"""
Idle Shutdown Service — auto-stops the vast.ai instance after inactivity.

Tracks the timestamp of the last API request. A background task checks
periodically and triggers shutdown when idle time exceeds the threshold.

Shutdown methods (in priority order):
1. Vast.ai API — cleanly stops the instance via their REST API
2. System shutdown — `shutdown -h now` as fallback
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class IdleShutdownService:
    """Monitors activity and triggers auto-shutdown on idle timeout."""

    def __init__(self):
        self._last_activity: float = time.time()
        self._shutdown_task: Optional[asyncio.Task] = None
        self._is_shutting_down: bool = False
        self._check_interval_seconds: int = 60  # Check every minute

    @property
    def idle_seconds(self) -> float:
        """Seconds since last API activity."""
        return time.time() - self._last_activity

    @property
    def idle_minutes(self) -> float:
        return self.idle_seconds / 60.0

    @property
    def shutdown_threshold_seconds(self) -> float:
        return settings.idle_shutdown_minutes * 60.0

    @property
    def time_until_shutdown_seconds(self) -> float:
        """Seconds remaining before auto-shutdown triggers. Negative = overdue."""
        return self.shutdown_threshold_seconds - self.idle_seconds

    def touch(self):
        """Record API activity — resets the idle timer."""
        self._last_activity = time.time()

    def get_status(self) -> dict:
        """Return current idle/shutdown status."""
        return {
            "enabled": settings.idle_shutdown_enabled,
            "idle_minutes": round(self.idle_minutes, 1),
            "shutdown_after_minutes": settings.idle_shutdown_minutes,
            "minutes_until_shutdown": max(0, round(self.time_until_shutdown_seconds / 60.0, 1)),
            "is_shutting_down": self._is_shutting_down,
        }

    async def start(self):
        """Start the background idle monitor."""
        if not settings.idle_shutdown_enabled:
            logger.info("Idle auto-shutdown is DISABLED")
            return

        logger.info(
            f"Idle auto-shutdown ENABLED — instance will stop after "
            f"{settings.idle_shutdown_minutes} minutes of inactivity"
        )
        self._shutdown_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop the background monitor."""
        if self._shutdown_task and not self._shutdown_task.done():
            self._shutdown_task.cancel()
            try:
                await self._shutdown_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        """Background loop that checks idle time and triggers shutdown."""
        while True:
            try:
                await asyncio.sleep(self._check_interval_seconds)

                if self.idle_seconds >= self.shutdown_threshold_seconds:
                    logger.warning(
                        f"Instance idle for {self.idle_minutes:.0f} minutes "
                        f"(threshold: {settings.idle_shutdown_minutes} min). "
                        f"Initiating shutdown..."
                    )
                    await self._execute_shutdown()
                    return

                # Log remaining time every 5 minutes
                remaining = self.time_until_shutdown_seconds / 60.0
                if int(self.idle_seconds) % 300 < self._check_interval_seconds:
                    logger.info(
                        f"Idle monitor: {self.idle_minutes:.0f}m idle, "
                        f"{remaining:.0f}m until auto-shutdown"
                    )

            except asyncio.CancelledError:
                logger.info("Idle monitor cancelled")
                return
            except Exception as e:
                logger.error(f"Idle monitor error: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def _execute_shutdown(self):
        """Shut down the instance."""
        self._is_shutting_down = True

        # Method 1: Vast.ai API
        if settings.vastai_api_key:
            instance_id = settings.vastai_instance_id or self._detect_instance_id()
            if instance_id:
                logger.info(f"Stopping vast.ai instance {instance_id} via API...")
                success = await self._stop_via_vastai(instance_id)
                if success:
                    return

        # Method 2: System shutdown (works in any environment)
        logger.info("Executing system shutdown...")
        try:
            subprocess.run(["shutdown", "-h", "now"], check=False)
        except Exception as e:
            logger.error(f"System shutdown failed: {e}")
            # Last resort — just exit the process, vast.ai will see it as stopped
            logger.info("Forcing process exit...")
            os._exit(0)

    async def _stop_via_vastai(self, instance_id: str) -> bool:
        """Stop the instance via vast.ai REST API."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.put(
                    f"https://console.vast.ai/api/v0/instances/{instance_id}/",
                    headers={"Authorization": f"Bearer {settings.vastai_api_key}"},
                    json={"state": "stopped"},
                )
                if resp.status_code == 200:
                    logger.info(f"Vast.ai instance {instance_id} stop request successful")
                    return True
                else:
                    logger.error(
                        f"Vast.ai API returned {resp.status_code}: {resp.text}"
                    )
                    return False
        except Exception as e:
            logger.error(f"Vast.ai API call failed: {e}")
            return False

    @staticmethod
    def _detect_instance_id() -> Optional[str]:
        """Try to auto-detect the vast.ai instance ID from environment."""
        # Vast.ai sets CONTAINER_ID or similar env vars
        for var in ["VAST_CONTAINERLABEL", "CONTAINER_ID", "HOSTNAME"]:
            value = os.environ.get(var)
            if value:
                logger.info(f"Auto-detected instance ID from {var}: {value}")
                return value
        return None

    async def force_shutdown(self):
        """Manually trigger immediate shutdown."""
        logger.warning("Manual shutdown requested via API!")
        await self._execute_shutdown()


# Global singleton
idle_shutdown_service = IdleShutdownService()
