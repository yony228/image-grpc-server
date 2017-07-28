#! /usr/bin/env python
# -*- coding: utf-8 -*-
import grpc
import time
from concurrent import futures
import sys
from train import cvt
from train import train
from deploy import image_saved_model
from train.utils import mysql_util
import multiprocessing
import logging as _logging
import image_train_service_pb2
import image_train_service_pb2_grpc
import uuid
import Queue
import threading

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '[::]'
_PORT = '50051'

global train_handler


def invoke_format(prefix_str, t=None):
    if t:
        t.join()
    try:
        print '----------------------format begin-----------------------'
        cvt.main(prefix_str)
    except BaseException:
        print BaseException.message
        return
    else:
        return


def invoke_train(prefix_str, step_num=5000, t=None):
    t.join()
    print '----------------------train begin-----------------------'
    train.FLAGS.__setattr__("max_number_of_steps", step_num)
    train.FLAGS.__setattr__("train_dir", '/data/jg/image/train/tmp/ckpt/' + prefix_str)
    train.main(prefix_str)
    return


def invoke_transfer(prefix_str, version, t):
    t.join()
    print '----------------------transfer begin-----------------------'
    # image_saved_model.FLAGS.__setattr__("checkpoint_dir", 'models/scenes/')
    image_saved_model.FLAGS.__setattr__("checkpoint_dir", '/data/jg/image/train/tmp/ckpt/' + prefix_str)
    image_saved_model.FLAGS.__setattr__("output_dir", "/data/jg/image/train/tmp/models/" + prefix_str)
    image_saved_model.FLAGS.__setattr__("model_version", version)
    image_saved_model.main()
    return


def invoke_image_train(prefix_str, step_num=5000, version=1, **kwargs):
    # step_num = prefix_str[1]
    # version = prefix_str[2]
    # prefix_str = prefix_str[0]
    try:
        db = mysql_util._get_mysql_db()
        if not db:
            raise ValueError('mysql connections is invalid')
        cur = db.cursor()
        sql = 'UPDATE trains SET status = %d WHERE train_no= "%s"' % (1, prefix_str)
        cur.execute(sql)
        db.commit()

        print '----------------------format begin-----------------------'
        cvt.FLAGS.__setattr__("cvt_path_prefix", "/data/jg/image/images/")
        cvt.FLAGS.__setattr__("cvt_num_shards", 2)
        cvt.main(prefix_str)
        print '----------------------train begin-----------------------'
        train.FLAGS.__setattr__("max_number_of_steps", step_num)
        train.FLAGS.__setattr__("train_dir", '/data/jg/image/train/tmp/ckpt/' + prefix_str)
        train.main(prefix_str)
        print '----------------------transfer begin-----------------------'
        image_saved_model.FLAGS.__setattr__("checkpoint_dir", '/data/jg/image/train/tmp/ckpt/' + prefix_str)
        image_saved_model.FLAGS.__setattr__("labels_dir", '/data/jg/image/train/tmp/tfrecord/' + prefix_str)
        image_saved_model.FLAGS.__setattr__("output_dir", "/data/jg/image/train/tmp/models/" + prefix_str)
        image_saved_model.FLAGS.__setattr__("model_version", version)
        image_saved_model.out_def(step_num)

        sql = 'UPDATE trains SET status = %d WHERE train_no= "%s"' % (2, prefix_str)
        cur.execute(sql)
        db.commit()
    except Exception, e:
        print Exception, ":", e
        sql = 'UPDATE trains SET status = %d WHERE train_no= "%s"' % (-1, prefix_str)
        cur.execute(sql)
        db.commit()
        return 0
    else:
        return 1
    finally:
        if kwargs.has_key('is_finish'):
            kwargs['is_finish'].set()
        if db:
            db.close()


def invoke_test(prefix_str, step_num=5000, version=1, **kwargs):
    for i in range(0, 180):
        time.sleep(1)
    if kwargs.has_key('is_finish'):
        kwargs['is_finish'].set()


class MultiThreadHandler:
    def __init__(self):
        self.pool = futures.ThreadPoolExecutor(max_workers=2)
        self.strategies = {}

    def add_strategy(self, strategy_builder):
        self.strategies[strategy_builder.strategy_name] = strategy_builder.get_fun_()

    def exec_strategy(self, strategy_name, args):
        self.pool.submit(self.strategies[strategy_name], args)


class MultiProcessorHandler:
    def __init__(self):
        self.strategies = {}
        self.map = {}

    def wait_for_final(self):
        self.pool.close()
        self.pool.join()

    def exec_strategy(self, strategy_name, args):
        is_finish = multiprocessing.Event()
        p = multiprocessing.Process(target=self.strategies[strategy_name], args=args + (is_finish,))
        p.start()

        key = uuid.uuid1()
        self.map[key] = (p, is_finish)
        return key

    def add_strategy(self, strategy_builder):
        self.strategies[strategy_builder.strategy_name] = strategy_builder.get_fun_()

    def clean(self):
        for key in list(self.map):
            if self.map[key][1].is_set() or (not self.map[key][0].is_alive()):
                self.map[key][0].terminate()
                self.map[key][0].join()
                self.map.pop(key)


class MultiProcessorPoolHandler:
    def __init__(self, parallel_size):
        self.strategies = {}
        self.parallel_size = multiprocessing.cpu_count() if not parallel_size else parallel_size
        self.max_size = 10 if 10 > self.parallel_size else (self.parallel_size * 2)
        self.waiting_jobs = Queue.PriorityQueue(self.max_size - self.parallel_size)
        self.running_jobs = {}
        self.start_thread = threading.Thread(target=MultiProcessorPoolHandler.start_thread_fun, name='start_thread',
                                             args=(self,))
        self.clean_thread = threading.Thread(target=MultiProcessorPoolHandler.clean_thread_fun, name='clean_thread',
                                             args=(self,))
        self.start_thread.start()
        self.clean_thread.start()

    def start_thread_fun(self):
        while True:
            if not self.waiting_jobs.empty() and len(self.running_jobs) < self.parallel_size:
                next_job = self.waiting_jobs.get()
                is_finish = multiprocessing.Event()
                p = multiprocessing.Process(target=next_job[1], args=next_job[2], kwargs={'is_finish': is_finish})
                p.start()
                self.running_jobs[next_job[0]] = (p, is_finish)
                self.waiting_jobs.task_done()
            time.sleep(5)

    def clean_thread_fun(self):
        while True:
            time.sleep(5)
            self._clean_()

    def add_strategy(self, strategy_builder):
        self.strategies[strategy_builder.strategy_name] = strategy_builder.get_fun_()

    def exec_strategy(self, strategy_name, args):
        try:
            key = None
            message = None
            if not self.waiting_jobs.full():
                uid = uuid.uuid1()
                self.waiting_jobs.put((uid, self.strategies[strategy_name], args,), timeout=1)
                key = uid
        except Exception, e:
            message = e.message
            print e.message
        finally:
            return (key, message)

    def kill_job(self, key):
        for i in self.waiting_jobs:
            if i is Queue:
                pass
        for i in self.running_jobs:
            pass

    def _clean_(self):
        for key in list(self.running_jobs):
            if self.running_jobs[key][1].is_set() or (not self.running_jobs[key][0].is_alive()):
                self.running_jobs[key][0].terminate()
                self.running_jobs[key][0].join()
                self.running_jobs.pop(key)


class OriginalMultiProcessorPoolHandler:
    def __init__(self):
        self.pool = multiprocessing.Pool(processes=2)  # multiprocessing.cpu_count()
        self.strategies = {}

    def wait_for_final(self):
        self.pool.close()
        self.pool.join()

    def exec_strategy(self, strategy_name, args):
        self.pool.apply_async(self.strategies[strategy_name], args)

    def add_strategy(self, strategy_builder):
        self.strategies[strategy_builder.strategy_name] = strategy_builder.get_fun_()


class StrategyBuilder:
    def __init__(self, strategy_name, fun, wrapper, *wrapper_args, **wrapper_kwargs):
        self.strategy_name = strategy_name
        self.fun = fun
        self.wrapper = wrapper
        self.wrapper_args = wrapper_args
        self.wrapper_kwargs = wrapper_kwargs

    def get_fun_(self):
        if self.wrapper:
            return self.wrapper(self.fun, *self.wrapper_args, **self.wrapper_kwargs)
        else:
            return self.fun


class LogWrap:
    def __init__(self, func, logfile_type='static', logfile_path='/log.txt', *wrapper_args, **wrapper_kwargs):
        self.func = func
        self.wrapper_args = wrapper_args
        self.wrapper_kwargs = wrapper_kwargs
        self.logfile_type = logfile_type
        self.logfile_path = logfile_path

    def __call__(self, *args, **kwargs):
        if self.logfile_type == 'static':
            for arg in self.wrapper_args:
                self.logfile_path = arg
            for arg in self.wrapper_kwargs:
                self.logfile_path = arg

        elif self.logfile_type == 'dynamic':
            for arg in self.wrapper_args:
                if isinstance(arg, int) and abs(arg) <= len(args):
                    self.logfile_path = args[arg]
            for arg in self.wrapper_kwargs:
                if isinstance(self.wrapper_kwargs[arg], int) and abs(self.wrapper_kwargs[arg]) <= len(args):
                    self.logfile_path = args[self.wrapper_kwargs[arg]]

        saved_std_out = sys.stdout
        saved_std_err = sys.stderr
        import os
        if not os.path.exists(os.path.dirname(self.logfile_path)):
            os.makedirs(os.path.dirname(self.logfile_path))
        with open(self.logfile_path, 'w+') as file_:
            sys.stdout = file_
            sys.stderr = file_
            _logger = _logging.getLogger('tensorflow')
            _handler = _logging.StreamHandler(file_)
            _handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
            _logger.addHandler(_handler)

            print 'LogWrap begin'
            self.func(*args[:-1 if self.logfile_type == 'dynamic' else None], **kwargs)
            print 'LogWrap end'
            sys.stdout = saved_std_out
            sys.stderr = saved_std_err
            _logger.removeHandler(_handler)
        return


class TrainImage(image_train_service_pb2_grpc.TrainServiceServicer):
    def __call__(self, *args, **kwargs):
        pass

    def DoTrain(self, request, context):
        train_no = request.train_no
        train_step = request.train_step
        train_test_percent = request.train_test_percent
        train_shard_num = request.train_shard_num

        key, message = train_handler.exec_strategy('image_train',
                                    (train_no, train_step,
                                     2,
                                     u'/data/jg/image/train/tmp/logs/' + train_no + u'/stdout.txt'
                                     )
                                    )
        return image_train_service_pb2.TrainResponse(key=str(key), message=message)


def server():
    reload(sys)
    sys.setdefaultencoding('utf-8')
    global train_handler
    # train_handler = MultiThreadHandler()
    train_handler = MultiProcessorPoolHandler(parallel_size=2)
    train_handler.add_strategy(
        StrategyBuilder('image_train',
                        invoke_image_train,
                        LogWrap,
                        logfile_type='dynamic', logfile_path_index=-1)
    )

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    image_train_service_pb2_grpc.add_TrainServiceServicer_to_server(TrainImage(), grpc_server)
    grpc_server.add_insecure_port(_HOST + ':' + _PORT)
    grpc_server.start()
    try:
        while True:
            print "loop begin -----"
            time.sleep(5)
    except KeyboardInterrupt:
        grpc_server.stop(0)


if __name__ == '__main__':
    server()
