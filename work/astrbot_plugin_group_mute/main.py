"""Safe QQ group mute controls for AstrBot's aiocqhttp/NapCat adapter."""

from __future__ import annotations

import re
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core.message.components import At
from astrbot.core.star.filter.command import GreedyStr


_DURATION_RE = re.compile(r"(?:^|\s)(\d+)\s*(?:分钟|分|m|min|mins)\s*$", re.I)


class GroupMutePlugin(Star):
    """Restrict group mute actions to QQ administrators or an explicit whitelist."""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

    @filter.command("禁言")
    async def mute(self, event: AstrMessageEvent, args: GreedyStr):
        """禁言 @成员 [分钟]，例如：禁言 @小明 10分钟。"""
        message = await self._mute(event, str(args), unmute=False)
        yield event.plain_result(message)
        event.stop_event()

    @filter.command("解除禁言", alias={"解禁"})
    async def unmute(self, event: AstrMessageEvent, args: GreedyStr):
        """解除禁言 @成员。"""
        message = await self._mute(event, str(args), unmute=True)
        yield event.plain_result(message)
        event.stop_event()

    async def _mute(self, event: AstrMessageEvent, args: str, *, unmute: bool) -> str:
        if event.get_platform_name() != "aiocqhttp":
            return "该功能仅支持 QQ/NapCat（aiocqhttp）平台。"
        group_id = event.get_group_id()
        if not group_id:
            return "请在 QQ 群内使用此命令。"
        if not await self._is_authorized(event, group_id):
            return "权限不足：仅群主、群管理员或插件白名单成员可执行。"

        target_id = self._mentioned_user_id(event)
        if target_id is None:
            usage = "解除禁言 @成员" if unmute else "禁言 @成员 10分钟"
            return f"请 @ 需要操作的群成员。用法：{usage}"
        if target_id == str(event.get_sender_id()):
            return "为避免误操作，不能对自己执行该命令。"

        if unmute:
            duration = 0
        else:
            try:
                duration = self._parse_duration(args)
            except ValueError as exc:
                return str(exc)

        try:
            await self._call_action(
                event,
                "set_group_ban",
                group_id=int(group_id),
                user_id=int(target_id),
                duration=duration,
            )
        except Exception as exc:
            logger.exception("set_group_ban failed: group=%s target=%s", group_id, target_id)
            return f"操作失败：{self._safe_error(exc)}。请确认机器人是本群管理员，且目标不是群主或权限更高的管理员。"

        action = "已解除禁言" if unmute else f"已禁言 {duration // 60} 分钟"
        logger.info("group mute action: operator=%s group=%s target=%s duration=%s", event.get_sender_id(), group_id, target_id, duration)
        return f"{action}。"

    async def _is_authorized(self, event: AstrMessageEvent, group_id: str) -> bool:
        operator = str(event.get_sender_id())
        whitelist = {str(item).strip() for item in self.config.get("admin_qqs", []) if str(item).strip()}
        if operator in whitelist:
            return True
        if not self.config.get("allow_group_admins", True):
            return False
        try:
            member = await self._call_action(
                event,
                "get_group_member_info",
                group_id=int(group_id),
                user_id=int(operator),
                no_cache=False,
            )
            return isinstance(member, dict) and member.get("role") in {"owner", "admin"}
        except Exception:
            logger.exception("could not verify group-manager role for %s", operator)
            return False

    def _parse_duration(self, args: str) -> int:
        match = _DURATION_RE.search(args.strip())
        if not match:
            default = self._bounded_minutes(self.config.get("default_duration_minutes", 10))
            return default * 60
        minutes = self._bounded_minutes(int(match.group(1)))
        if minutes <= 0:
            raise ValueError("禁言时长必须大于 0 分钟。")
        return minutes * 60

    def _bounded_minutes(self, raw: object) -> int:
        try:
            minutes = int(raw)
        except (TypeError, ValueError):
            minutes = 10
        maximum = int(self.config.get("max_duration_minutes", 1440) or 0)
        if maximum > 0:
            minutes = min(minutes, maximum)
        return max(minutes, 1)

    @staticmethod
    def _mentioned_user_id(event: AstrMessageEvent) -> str | None:
        chain = getattr(getattr(event, "message_obj", None), "message", [])
        for component in chain:
            if isinstance(component, At):
                qq = str(getattr(component, "qq", ""))
                if qq.isdigit():
                    return qq
        return None

    @staticmethod
    async def _call_action(event: AstrMessageEvent, action: str, **payload: Any) -> Any:
        bot = getattr(event, "bot", None)
        call_action = getattr(bot, "call_action", None)
        if not callable(call_action):
            raise RuntimeError("当前 OneBot 客户端不支持 call_action")
        return await call_action(action, **payload)

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        text = str(exc).replace("\n", " ").strip()
        return text[:120] if text else exc.__class__.__name__
