"""Group games plus a multiplayer truth-or-dare state machine."""

from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass, field

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star


@dataclass
class BombGame:
    low: int
    high: int
    answer: int


@dataclass
class RouletteGame:
    chambers: int
    bullet: int
    current: int = 0


@dataclass
class PartyGame:
    participant_count: int
    mode: str
    names: dict[str, str] = field(default_factory=dict)
    results: dict[str, int | str] = field(default_factory=dict)
    eligible: set[str] | None = None
    loser_id: str | None = None
    penalty_kind: str | None = None
    refreshes_used: int = 0


class GroupGamesPlugin(Star):
    """Run explicit commands; call the LLM only after the loser chooses a task."""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._bomb_games: dict[str, BombGame] = {}
        self._roulette_games: dict[str, RouletteGame] = {}
        self._party_games: dict[str, PartyGame] = {}
        self._lock = asyncio.Lock()

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def handle_game_command(self, event: AstrMessageEvent):
        """群小游戏入口：无需 @机器人，也不触发普通聊天 LLM。

        可用指令：/开始（俄罗斯转盘）、shot、/开始 炸弹 1 100、/50、
        /开始 真心话大冒险 <人数> 骰子（参与者 /投）、
        /开始 真心话大冒险 <人数> 猜拳（参与者 /cut）、/break。
        真心话大冒险发生平局时，仅平局者按提示重抽；唯一失败者可选择
        /真心话 或 /大冒险，并可 /刷新两次。仅最终生成真心话或大冒险题目时调用 LLM。
        """
        group_id = str(event.get_group_id() or "")
        if not group_id:
            return

        text = event.get_message_str().strip()
        if not self._is_game_command(text):
            return
        event.should_call_llm(False)
        event.stop_event()
        operator_id = str(event.get_sender_id())
        operator_name = event.get_sender_name() or operator_id

        if text in {"/真心话", "/大冒险", "/刷新"}:
            reply = await self._choose_or_refresh_penalty(
                event,
                group_id,
                operator_id,
                text,
            )
        else:
            async with self._lock:
                reply = self._handle(
                    group_id,
                    operator_id,
                    operator_name,
                    text,
                )
        yield event.plain_result(reply)

    @staticmethod
    def _is_game_command(text: str) -> bool:
        if text in {"/break", "shot", "/shot", "/投", "/cut", "/真心话", "/大冒险", "/刷新", "/小游戏帮助"}:
            return True
        if text.startswith("/") and text[1:].lstrip("+-").isdigit():
            return True
        return bool(re.match(r"^/开始(?:\s|$)", text))

    def _handle(
        self,
        group_id: str,
        operator_id: str,
        operator_name: str,
        text: str,
    ) -> str:
        if text == "/小游戏帮助":
            return self._help_text()
        if text == "/break":
            return self._break(group_id)
        if text in {"shot", "/shot"}:
            return self._fire(group_id)
        if text.startswith("/") and text[1:].lstrip("+-").isdigit():
            return self._guess(group_id, text[1:])
        if text == "/投":
            return self._party_action(group_id, operator_id, operator_name, "dice")
        if text == "/cut":
            return self._party_action(group_id, operator_id, operator_name, "rps")
        return self._start(group_id, text.removeprefix("/开始").strip())

    def _start(self, group_id: str, args: str) -> str:
        tokens = args.split()
        if not tokens or tokens[0] in {"转盘", "俄罗斯转盘", "roulette"}:
            self._clear_group_games(group_id)
            chambers = self._chambers()
            self._roulette_games[group_id] = RouletteGame(chambers, random.randrange(chambers))
            return f"🎲 转盘开始（{chambers} 格，纯娱乐）。发送 shot。"
        if tokens[0] in {"炸弹", "数字炸弹", "bomb"}:
            try:
                low, high = self._parse_range(tokens[1:])
            except ValueError as exc:
                return str(exc)
            self._clear_group_games(group_id)
            self._bomb_games[group_id] = BombGame(low, high, random.randint(low, high))
            return f"💣 数字炸弹：{low}～{high}。直接发送 /数字，例如 /50。"
        if tokens[0] in {"真心话大冒险", "真心话", "大冒险", "td"}:
            return self._start_party(group_id, tokens[1:])
        return "未知游戏。发送 /小游戏帮助 查看玩法。"

    def _start_party(self, group_id: str, tokens: list[str]) -> str:
        if len(tokens) != 2:
            return (
                "用法：/开始 真心话大冒险 4 骰子\n"
                "或：/开始 真心话大冒险 4 猜拳"
            )
        try:
            count = int(tokens[0])
        except ValueError:
            return "参与人数必须是 2～20 的整数。"
        if not 2 <= count <= 20:
            return "参与人数必须在 2～20 之间。"
        mode_token = tokens[1].lower()
        if mode_token in {"骰子", "色子", "dice", "投"}:
            mode = "dice"
            instruction = "参与者发送 /投"
        elif mode_token in {"猜拳", "剪刀石头布", "rps", "cut"}:
            mode = "rps"
            instruction = "参与者发送 /cut"
        else:
            return "决定方式只能选择“骰子”或“猜拳”。"
        self._clear_group_games(group_id)
        self._party_games[group_id] = PartyGame(count, mode)
        return f"🎭 真心话大冒险开始：{count} 人，{tokens[1]}决定胜负。{instruction}。"

    def _party_action(
        self,
        group_id: str,
        operator_id: str,
        operator_name: str,
        requested_mode: str,
    ) -> str:
        game = self._party_games.get(group_id)
        if game is None:
            return "未开始真心话大冒险。发送 /小游戏帮助 查看用法。"
        if game.mode != requested_mode:
            command = "/投" if game.mode == "dice" else "/cut"
            return f"本局使用另一种决定方式，请发送 {command}。"
        if game.loser_id is not None:
            loser_name = game.names.get(game.loser_id, game.loser_id)
            if game.penalty_kind is None:
                return f"等待 {loser_name} 选择 /真心话 或 /大冒险。"
            remaining = 2 - game.refreshes_used
            return f"题目已生成；{loser_name} 还可 /刷新 {remaining} 次。"
        if game.eligible is not None and operator_id not in game.eligible:
            tied = "、".join(game.names[user_id] for user_id in game.eligible)
            return f"本轮仅平局者参与：{tied}。"
        if operator_id in game.results:
            return f"{operator_name} 本轮已经抽过。"

        expected = len(game.eligible) if game.eligible is not None else game.participant_count
        if game.eligible is None and len(game.names) >= game.participant_count:
            return "本局参与人数已满。"
        game.names.setdefault(operator_id, operator_name)

        if game.mode == "dice":
            result: int | str = random.randint(1, self._dice_sides())
            action_text = f"🎲 {operator_name}：{result} 点"
        else:
            result = random.choice(("石头", "剪刀", "布"))
            action_text = f"✂️ {operator_name}：{result}"
        game.results[operator_id] = result

        if len(game.results) < expected:
            return f"{action_text}（{len(game.results)}/{expected}）"
        resolution = self._resolve_party_round(game)
        return f"{action_text}\n{resolution}"

    def _resolve_party_round(self, game: PartyGame) -> str:
        details = "；".join(
            f"{game.names[user_id]}={value}"
            for user_id, value in game.results.items()
        )
        if game.mode == "dice":
            losers = self._dice_losers(game.results)
        else:
            losers = self._rps_losers(game.results)

        if len(losers) > 1:
            game.eligible = set(losers)
            game.results.clear()
            names = "、".join(game.names[user_id] for user_id in losers)
            command = "/投" if game.mode == "dice" else "/cut"
            return f"本轮：{details}\n平局：{names}。仅以上成员再次发送 {command}。"

        loser_id = losers[0]
        game.loser_id = loser_id
        loser_name = game.names[loser_id]
        return f"本轮：{details}\n失败者：{loser_name}。请选择 /真心话 或 /大冒险。"

    @classmethod
    def _dice_losers(cls, results: dict[str, int | str]) -> list[str]:
        ids = list(results)
        scores = dict.fromkeys(ids, 0)
        for index, left_id in enumerate(ids):
            left = int(results[left_id])
            for right_id in ids[index + 1 :]:
                right = int(results[right_id])
                comparison = cls._compare_dice(left, right)
                scores[left_id] += comparison
                scores[right_id] -= comparison
        lowest = min(scores.values())
        return [user_id for user_id in ids if scores[user_id] == lowest]

    @staticmethod
    def _compare_dice(left: int, right: int) -> int:
        if left == right:
            return 0
        if left == 1 and right == 6:
            return 1
        if left == 6 and right == 1:
            return -1
        return 1 if left > right else -1

    @staticmethod
    def _rps_losers(results: dict[str, int | str]) -> list[str]:
        moves = {str(value) for value in results.values()}
        ids = list(results)
        if len(moves) in {1, 3}:
            return ids
        if moves == {"石头", "剪刀"}:
            losing_move = "剪刀"
        elif moves == {"剪刀", "布"}:
            losing_move = "布"
        else:
            losing_move = "石头"
        return [user_id for user_id, move in results.items() if move == losing_move]

    async def _choose_or_refresh_penalty(
        self,
        event: AstrMessageEvent,
        group_id: str,
        operator_id: str,
        command: str,
    ) -> str:
        async with self._lock:
            game = self._party_games.get(group_id)
            if game is None or game.loser_id is None:
                return "尚未决出失败者。"
            loser_name = game.names.get(game.loser_id, game.loser_id)
            if operator_id != game.loser_id:
                return f"仅失败者 {loser_name} 可以选择。"
            if command == "/刷新":
                if game.penalty_kind is None:
                    return "请先选择 /真心话 或 /大冒险。"
                if game.refreshes_used >= 2:
                    return "本轮两次刷新机会已经用完。"
                game.refreshes_used += 1
                refresh_number = game.refreshes_used
                is_truth = game.penalty_kind == "truth"
            else:
                if game.penalty_kind is not None:
                    remaining = 2 - game.refreshes_used
                    return f"本轮已选择题目；如需更换请发送 /刷新（剩余 {remaining} 次）。"
                is_truth = command == "/真心话"
                game.penalty_kind = "truth" if is_truth else "dare"
                refresh_number = 0
        question = await self._generate_truth_or_dare(event, is_truth)
        kind = "💬 真心话" if is_truth else "🎯 大冒险"
        if refresh_number:
            remaining = 2 - refresh_number
            return f"🔄 已刷新（{refresh_number}/2，剩余 {remaining} 次）\n{kind}：{question}"
        return f"{loser_name} 选择了{kind[2:]}。\n{kind}：{question}\n可发送 /刷新，更换题目（2次）。"

    async def _generate_truth_or_dare(
        self,
        event: AstrMessageEvent,
        is_truth: bool,
    ) -> str:
        kind = "真心话问题" if is_truth else "大冒险任务"
        system_prompt = (
            "生成适合普通QQ群多人聚会的真心话大冒险题目。"
            "题目可以稍有私密感和社交压力，可以涉及恋爱经历、前任分手原因、暗恋、"
            "心动对象、吃醋和尴尬往事，但不得要求说出真实姓名、联系方式、住址或其他"
            "可识别个人的信息。禁止露骨色情、强迫骚扰、歧视、危险或违法行为、自伤、"
            "饮酒惩罚和金钱要求。题目应简短、有趣、可拒绝，不要羞辱参与者。"
            "只输出一道题目，不要解释，不要编号。"
        )
        try:
            provider_id = await self.context.get_current_chat_provider_id(
                event.unified_msg_origin
            )
            response = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=f"生成一道{kind}，控制在50个汉字以内。",
                system_prompt=system_prompt,
                max_tokens=100,
                temperature=0.9,
            )
            question = response.completion_text.strip().strip('"“”')
            if not question:
                raise RuntimeError("模型返回空内容")
            return question.replace("\n", " ")[:100]
        except Exception:
            logger.exception("truth-or-dare LLM generation failed")
            return random.choice(
                [
                    "最近做过最有成就感的一件事是什么？",
                    "分享一个很少有人知道的小习惯。",
                ]
                if is_truth
                else [
                    "用三个表情描述今天的心情。",
                    "模仿群里常见的说话方式发一句话。",
                ]
            )

    def _break(self, group_id: str) -> str:
        existed = self._clear_group_games(group_id)
        return "本群小游戏已结束。" if existed else "当前群没有进行中的小游戏。"

    def _fire(self, group_id: str) -> str:
        game = self._roulette_games.get(group_id)
        if game is None:
            return "未开始转盘。发送 /开始。"
        if game.current == game.bullet:
            self._roulette_games.pop(group_id, None)
            return "💥 砰！本局结束。"
        game.current += 1
        return f"咔哒，安全。剩 {game.chambers - game.current} 格；下一位 shot。"

    def _guess(self, group_id: str, raw_guess: str) -> str:
        game = self._bomb_games.get(group_id)
        if game is None:
            return "未开始数字炸弹。使用 /开始 炸弹 1 100。"
        guess = int(raw_guess)
        if not game.low <= guess <= game.high:
            return f"范围：{game.low}～{game.high}。"
        if guess == game.answer:
            self._bomb_games.pop(group_id, None)
            return f"💥 {guess}，炸弹！本局结束。"
        if guess < game.answer:
            game.low = guess + 1
        else:
            game.high = guess - 1
        return f"安全，范围缩小为 {game.low}～{game.high}。"

    def _parse_range(self, tokens: list[str]) -> tuple[int, int]:
        if not tokens:
            return 1, 100
        if len(tokens) != 2:
            raise ValueError("用法：/开始 炸弹 1 100")
        try:
            low, high = int(tokens[0]), int(tokens[1])
        except ValueError as exc:
            raise ValueError("范围必须是整数，例如 /开始 炸弹 1 100") from exc
        maximum = int(self.config.get("number_bomb_max_range", 1000) or 1000)
        if low >= high:
            raise ValueError("最小值必须小于最大值。")
        if high > maximum:
            raise ValueError(f"最大值不能超过 {maximum}。")
        return low, high

    def _chambers(self) -> int:
        try:
            return min(max(int(self.config.get("roulette_chambers", 6)), 2), 12)
        except (TypeError, ValueError):
            return 6

    def _dice_sides(self) -> int:
        try:
            return min(max(int(self.config.get("dice_sides", 6)), 2), 100)
        except (TypeError, ValueError):
            return 6

    def _clear_group_games(self, group_id: str) -> bool:
        removed = False
        for games in (self._bomb_games, self._roulette_games, self._party_games):
            removed = games.pop(group_id, None) is not None or removed
        return removed

    @staticmethod
    def _help_text() -> str:
        return (
            "【群小游戏】\n"
            "/开始：俄罗斯转盘；shot：进行回合\n"
            "/开始 炸弹 1 100；/50：猜数字\n"
            "/开始 真心话大冒险 4 骰子；参与者 /投\n"
            "/开始 真心话大冒险 4 猜拳；参与者 /cut\n"
            "平局者按提示重抽；失败者选择 /真心话 或 /大冒险\n"
            "题目不满意可 /刷新，每轮最多两次\n"
            "/break：结束当前游戏"
        )
