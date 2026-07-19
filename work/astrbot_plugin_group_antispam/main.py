"""Configurable per-user group flood detection for AstrBot QQ bots."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from time import monotonic
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star


@dataclass
class Violation:
    count: int
    last_at: float


class GroupAntiSpamPlugin(Star):
    """Detect fast, repeated messages per QQ user and apply configurable penalties."""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._messages: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._violations: dict[tuple[str, str], Violation] = {}
        self._cooldowns: dict[tuple[str, str], float] = {}
        self._lock = asyncio.Lock()

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def detect_flood(self, event: AstrMessageEvent):
        """自动群防刷屏：按 QQ 用户 ID 独立统计，不需要命令或 @机器人。

        在配置的时间窗口内，某用户的消息数达到阈值即触发处罚。可在插件配置页调整阈值、
        窗口、禁言时长、升级处罚、冷却时间、白名单及是否豁免群管理。仅支持 QQ/NapCat 的
        aiocqhttp 平台；正常消息不会被拦截，也不会调用 LLM。
        """
        if not self.config.get("enabled", True):
            return
        if event.get_platform_name() != "aiocqhttp":
            return

        group_id = str(event.get_group_id() or "")
        user_id = str(event.get_sender_id() or "")
        if not group_id or not user_id or self._is_whitelisted(user_id):
            return

        now = monotonic()
        key = (group_id, user_id)
        async with self._lock:
            if now < self._cooldowns.get(key, 0):
                return
            messages = self._messages[key]
            window = self._window_seconds()
            while messages and now - messages[0] > window:
                messages.popleft()
            messages.append(now)
            if len(messages) < self._message_threshold():
                self._prune_state(now)
                return
            messages.clear()
            self._cooldowns[key] = now + self._cooldown_seconds()

        if await self._is_exempt_group_admin(event, group_id, user_id):
            return

        duration = await self._record_violation_and_duration(key, now)
        try:
            await self._call_action(
                event,
                "set_group_ban",
                group_id=int(group_id),
                user_id=int(user_id),
                duration=duration,
            )
        except Exception as exc:
            logger.exception("anti-spam mute failed: group=%s user=%s", group_id, user_id)
            async with self._lock:
                self._cooldowns.pop(key, None)
            return

        minutes = max(1, duration // 60)
        name = event.get_sender_name() or user_id
        logger.info(
            "anti-spam action: group=%s user=%s duration=%ss",
            group_id,
            user_id,
            duration,
        )
        event.should_call_llm(False)
        event.stop_event()
        yield event.plain_result(
            self._notice_text().format(name=name, user_id=user_id, minutes=minutes)
        )

    async def _record_violation_and_duration(
        self, key: tuple[str, str], now: float
    ) -> int:
        async with self._lock:
            prior = self._violations.get(key)
            if prior and now - prior.last_at <= self._escalation_window_seconds():
                count = prior.count + 1
            else:
                count = 1
            self._violations[key] = Violation(count=count, last_at=now)
        base = self._mute_minutes() * 60
        if not self.config.get("escalate_penalty", True):
            return base
        multiplier = self._bounded_int(self.config.get("escalation_multiplier", 2), 1, 10)
        maximum = self._max_mute_minutes() * 60
        return min(base * (multiplier ** (count - 1)), maximum)

    async def _is_exempt_group_admin(
        self, event: AstrMessageEvent, group_id: str, user_id: str
    ) -> bool:
        if not self.config.get("exempt_group_admins", True):
            return False
        try:
            member = await self._call_action(
                event,
                "get_group_member_info",
                group_id=int(group_id),
                user_id=int(user_id),
                no_cache=False,
            )
            return isinstance(member, dict) and member.get("role") in {"owner", "admin"}
        except Exception:
            logger.warning("anti-spam could not check group role: group=%s user=%s", group_id, user_id)
            return False

    def _is_whitelisted(self, user_id: str) -> bool:
        return user_id in {
            str(item).strip()
            for item in self.config.get("whitelist_qqs", [])
            if str(item).strip()
        }

    def _prune_state(self, now: float) -> None:
        expiry = max(self._window_seconds(), self._escalation_window_seconds()) * 2
        stale_messages = [
            key
            for key, messages in self._messages.items()
            if not messages or now - messages[-1] > expiry
        ]
        for key in stale_messages:
            self._messages.pop(key, None)
        stale = [key for key, violation in self._violations.items() if now - violation.last_at > expiry]
        for key in stale:
            self._violations.pop(key, None)
        expired_cooldowns = [key for key, until in self._cooldowns.items() if until <= now]
        for key in expired_cooldowns:
            self._cooldowns.pop(key, None)

    def _window_seconds(self) -> int:
        return self._bounded_int(self.config.get("window_seconds", 10), 1, 300)

    def _message_threshold(self) -> int:
        return self._bounded_int(self.config.get("message_threshold", 6), 2, 100)

    def _mute_minutes(self) -> int:
        return self._bounded_int(self.config.get("mute_minutes", 5), 1, self._max_mute_minutes())

    def _max_mute_minutes(self) -> int:
        return self._bounded_int(self.config.get("max_mute_minutes", 1440), 1, 43200)

    def _cooldown_seconds(self) -> int:
        return self._bounded_int(self.config.get("action_cooldown_seconds", 30), 0, 3600)

    def _escalation_window_seconds(self) -> int:
        minutes = self._bounded_int(self.config.get("escalation_window_minutes", 60), 1, 10080)
        return minutes * 60

    def _notice_text(self) -> str:
        text = str(
            self.config.get(
                "notice_template",
                "检测到 {name}（{user_id}）短时间内发送过多消息，已禁言 {minutes} 分钟。",
            )
        ).strip()
        return text or "检测到 {name}（{user_id}）刷屏，已禁言 {minutes} 分钟。"

    @staticmethod
    def _bounded_int(value: object, lower: int, upper: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = lower
        return min(max(number, lower), upper)

    @staticmethod
    async def _call_action(event: AstrMessageEvent, action: str, **payload: Any) -> Any:
        bot = getattr(event, "bot", None)
        call_action = getattr(bot, "call_action", None)
        if not callable(call_action):
            raise RuntimeError("当前 OneBot 客户端不支持 call_action")
        return await call_action(action, **payload)
