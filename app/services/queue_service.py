import asyncio
import logging
import uuid
import time
from enum import IntEnum
from typing import Any, Callable, Coroutine, Dict, Optional

from app.services.model_manager import model_manager, ModelType

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """
    Lower value = Higher priority.
    """
    TEXT = 0
    AUDIO = 1
    IMAGE = 2
    VIDEO = 3


class QueuedTask:
    """Represents a task waiting for a GPU slot."""

    def __init__(
        self,
        task_id: str,
        model_type: ModelType,
        priority: Priority,
        func: Callable[..., Coroutine[Any, Any, Any]],
        args: tuple,
        kwargs: dict,
    ):
        self.task_id = task_id
        self.model_type = model_type
        self.priority = priority
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.future = asyncio.Future()
        self.queued_at = time.time()

    def __lt__(self, other: "QueuedTask"):
        # Used by PriorityQueue
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.queued_at < other.queued_at


class QueueService:
    """
    Manages a prioritized queue of AI tasks and dispatches them to available GPUs.
    Only one heavy task is allowed per GPU at a time.
    """

    def __init__(self):
        self._queue = asyncio.PriorityQueue()
        # Per-GPU execution lock (ensures 1 heavy task per GPU)
        self._gpu_locks: Dict[int, asyncio.Lock] = {}
        self._is_running = False
        self._worker_tasks: list[asyncio.Task] = []

    def _init_gpu_locks(self):
        if not self._gpu_locks:
            for i in range(model_manager.device_count):
                self._gpu_locks[i] = asyncio.Lock()

    async def start(self):
        """Start the background workers that process the queue."""
        if self._is_running:
            return
            
        self._init_gpu_locks()
        self._is_running = True
        logger.info(f"QueueService starting with {len(self._gpu_locks)} GPU worker(s)")
        
        # Start a worker loop for each GPU
        for i in range(len(self._gpu_locks)):
            worker = asyncio.create_task(self._gpu_worker_loop(i))
            self._worker_tasks.append(worker)

    async def stop(self):
        """Stop all workers."""
        self._is_running = False
        for worker in self._worker_tasks:
            worker.cancel()
        
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            self._worker_tasks = []

    async def enqueue(
        self,
        model_type: ModelType,
        priority: Priority,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args,
        **kwargs
    ) -> Any:
        """
        Add a task to the queue and wait for its completion.
        """
        task_id = str(uuid.uuid4())
        task = QueuedTask(task_id, model_type, priority, func, args, kwargs)
        
        await self._queue.put(task)
        logger.info(f"Task {task_id} ({model_type.value}) enqueued with priority {priority.name}")
        
        # Wait for the worker to finish the task
        return await task.future

    async def _gpu_worker_loop(self, device_id: int):
        """
        Worker loop pinned to a specific GPU.
        It pulls tasks from the global queue but only executes if it's the "best" GPU
        or if other GPUs are busy.
        """
        logger.info(f"Worker for cuda:{device_id} started")
        
        while self._is_running:
            try:
                # Get next task from the priority queue
                task: QueuedTask = await self._queue.get()
                
                # Check if this is the "best" GPU for the task (where model is already loaded)
                # If we have multiple GPUs, we might want to let the GPU with the cached model
                # grab the task. But PriorityQueue.get() is destructive.
                # Simplify: This worker takes the task and runs it on ITS assigned GPU.
                
                async with self._gpu_locks[device_id]:
                    logger.info(f"Worker {device_id} executing task {task.task_id} ({task.model_type.value})")
                    try:
                        # Pass the device_id to the generation function
                        result = await task.func(*task.args, device_id=device_id, **task.kwargs)
                        task.future.set_result(result)
                    except Exception as e:
                        logger.error(f"Task {task.task_id} failed on GPU {device_id}: {e}", exc_info=True)
                        task.future.set_exception(e)
                    finally:
                        self._queue.task_done()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker loop {device_id} error: {e}", exc_info=True)
                await asyncio.sleep(1)


# Global singleton
queue_service = QueueService()
