from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import heapq
import numpy as np

Coord = Tuple[int, int]

# 动作编码：上 / 下 / 左 / 右
ACTIONS: Dict[int, Coord] = {
    0: (-1, 0),   # 上
    1: (1, 0),    # 下
    2: (0, -1),   # 左
    3: (0, 1),    # 右
}

# 可视化时用到的箭头
ACTION_ARROWS: Dict[int, str] = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→",
}


@dataclass
class GridConfig:
    """
    SSP 网格环境配置，支持：
      - 敌人位置进状态
      - 玩家血量进状态
      - 塔伤害模型
    """

    grid: List[str]

    # 基础代价 & 风险格
    step_cost: float = 1.0
    risk_extra_cost: float = 4.0
    p_ambush: float = 0.0
    ambush_penalty: float = 0.0
    seed: int = 42

    # 敌人相关
    use_enemy_state: bool = False
    enemy_start: Optional[Coord] = None
    enemy_chase: bool = True
    catch_penalty: float = 50.0

    # 血量相关
    use_hp_state: bool = False
    max_hp: int = 10
    risky_damage: int = 2
    ambush_damage: int = 3
    catch_damage: int = 4
    death_penalty: float = 50.0

    # 塔相关
    use_tower: bool = False          # 是否启用塔伤害
    tower_range: int = 3             # 塔攻击范围（曼哈顿距离）
    tower_damage: int = 4            # 塔每次攻击扣的 HP
    tower_penalty: float = 8.0       # 被塔打一次额外 cost（相当于“吃技能伤害”）


class SSPGrid:
    """
    支持四种状态结构：
      1) 仅位置:            s = (agent)
      2) 位置 + 敌人:        s = (agent, enemy)
      3) 位置 + HP:         s = (agent, hp)
      4) 位置 + 敌人 + HP:   s = (agent, enemy, hp)
    """

    def __init__(self, cfg: GridConfig):
        self.cfg = cfg
        self.grid = cfg.grid
        self.h = len(cfg.grid)
        self.w = len(cfg.grid[0]) if self.h > 0 else 0

        # 起点 / 终点
        self.start: Coord = self._find("S")
        assert self.start is not None, "Grid must contain a start 'S'."
        self.goal: Optional[Coord] = self._find("G")

        # 危险格 / 障碍格
        self.risky = {
            (i, j)
            for i in range(self.h)
            for j in range(self.w)
            if cfg.grid[i][j] == "R"
        }
        self.blocked = {
            (i, j)
            for i in range(self.h)
            for j in range(self.w)
            if cfg.grid[i][j] == "#"
        }

        # 塔位置
        self.towers = {
            (i, j)
            for i in range(self.h)
            for j in range(self.w)
            if cfg.grid[i][j] == "T"
        }

        # 敌人初始位置
        e_pos = cfg.enemy_start
        if e_pos is None:
            e_pos = self._find("E")
        self.enemy_start: Optional[Coord] = e_pos

        # 模式开关
        self.use_enemy_state = cfg.use_enemy_state
        self.use_hp_state = cfg.use_hp_state

        # 随机
        self._rng = np.random.RandomState(cfg.seed)

    # 基础工具函数
    def _find(self, ch: str) -> Optional[Coord]:
        for i in range(self.h):
            for j in range(self.w):
                if self.grid[i][j] == ch:
                    return (i, j)
        return None

    def in_bounds(self, s: Coord) -> bool:
        """是否在地图内且不是障碍。"""
        x, y = s
        return 0 <= x < self.h and 0 <= y < self.w and (x, y) not in self.blocked

    # 塔攻击判定
    def _tower_in_range(self, pos: Coord) -> bool:
        """是否有任意一座塔能打到该位置（简单：曼哈顿距离判断）"""
        if not self.cfg.use_tower or not self.towers:
            return False
        for tx, ty in self.towers:
            dist = abs(tx - pos[0]) + abs(ty - pos[1])
            if dist <= self.cfg.tower_range:
                return True
        return False

    # 单智能体 / 敌人 / HP
    def is_goal_single(self, s: Coord) -> bool:
        return self.goal is not None and s == self.goal

    def all_states_single(self):
        return [
            (i, j)
            for i in range(self.h)
            for j in range(self.w)
            if (i, j) not in self.blocked
        ]

    def _initial_enemy_pos(self) -> Coord:
        if self.enemy_start is not None and self.in_bounds(self.enemy_start):
            return self.enemy_start

        default = (self.h - 1, self.w - 1)
        if self.in_bounds(default):
            return default

        for i in range(self.h - 1, -1, -1):
            for j in range(self.w - 1, -1, -1):
                if (i, j) not in self.blocked:
                    return (i, j)
        return self.start

    def _initial_hp(self) -> int:
        return self.cfg.max_hp

    def _is_dead_hp(self, hp: int) -> bool:
        return hp <= 0

    # 终止判定（统一接口）
    def is_goal(self, s: Any) -> bool:
        # 模式 1：仅位置
        if not self.use_enemy_state and not self.use_hp_state:
            return self.is_goal_single(s)

        # 模式 2：位置 + 敌人
        if self.use_enemy_state and not self.use_hp_state:
            agent, enemy = s
            if self.goal is not None and agent == self.goal:
                return True
            if agent == enemy:
                return True
            return False

        # 模式 3：位置 + HP
        if not self.use_enemy_state and self.use_hp_state:
            agent, hp = s
            if self.goal is not None and agent == self.goal:
                return True
            if self._is_dead_hp(hp):
                return True
            return False

        # 模式 4：位置 + 敌人 + HP
        agent, enemy, hp = s
        if self.goal is not None and agent == self.goal:
            return True
        if self._is_dead_hp(hp):
            return True
        if agent == enemy:
            return True
        return False

    #  状态空间枚举
    def all_states(self):
        base = self.all_states_single()

        # 模式 1：仅位置
        if not self.use_enemy_state and not self.use_hp_state:
            return base

        # 模式 2：位置 + 敌人
        if self.use_enemy_state and not self.use_hp_state:
            return [(a, e) for a in base for e in base]

        # 模式 3：位置 + HP
        if not self.use_enemy_state and self.use_hp_state:
            states = []
            for a in base:
                for hp in range(0, self.cfg.max_hp + 1):
                    states.append((a, hp))
            return states

        # 模式 4：位置 + 敌人 + HP
        states = []
        for a in base:
            for e in base:
                for hp in range(0, self.cfg.max_hp + 1):
                    states.append((a, e, hp))
        return states

    # Reset / Move
    def reset(self):
        """
        返回初始状态，根据不同模式返回不同结构。
        """
        if not self.use_enemy_state and not self.use_hp_state:
            return self.start

        enemy_pos = self._initial_enemy_pos()

        if self.use_enemy_state and not self.use_hp_state:
            return (self.start, enemy_pos)

        hp0 = self._initial_hp()

        if not self.use_enemy_state and self.use_hp_state:
            return (self.start, hp0)

        return (self.start, enemy_pos, hp0)

    def _next_det_agent(self, agent: Coord, a: int) -> Coord:
        """确定性移动：撞墙则原地。"""
        dx, dy = ACTIONS[a]
        s2 = (agent[0] + dx, agent[1] + dy)
        if not self.in_bounds(s2):
            s2 = agent
        return s2

    def move(self, s: Coord, a: int) -> Coord:
        """
        给 viz.plot_path 用的简单一步移动（只处理坐标）。
        """
        return self._next_det_agent(s, a)

    # 期望 cost（给 VI 用）
    def _expected_cost_single(self, s: Coord, a: int) -> float:
        s2 = self._next_det_agent(s, a)
        c = self.cfg.step_cost

        if s2 in self.risky:
            c += self.cfg.risk_extra_cost
            if self.cfg.p_ambush > 0.0 and self.cfg.ambush_penalty > 0.0:
                c += self.cfg.p_ambush * self.cfg.ambush_penalty

        if self._tower_in_range(s2):
            c += self.cfg.tower_penalty

        return c

    def _expected_cost_with_enemy(self, s, a: int) -> float:
        agent, enemy = s
        agent_next = self._next_det_agent(agent, a)
        if self.cfg.enemy_chase:
            enemy_next = self.astar_next_step(enemy, agent_next)
        else:
            enemy_next = enemy

        c = self.cfg.step_cost

        if agent_next in self.risky:
            c += self.cfg.risk_extra_cost
            if self.cfg.p_ambush > 0.0 and self.cfg.ambush_penalty > 0.0:
                c += self.cfg.p_ambush * self.cfg.ambush_penalty

        if agent_next == enemy_next:
            c += self.cfg.catch_penalty

        if self._tower_in_range(agent_next):
            c += self.cfg.tower_penalty

        return c

    def _expected_cost_with_hp(self, s, a: int) -> float:
        agent, hp = s
        agent_next = self._next_det_agent(agent, a)

        c = self.cfg.step_cost
        dmg_exp = 0.0

        if agent_next in self.risky:
            c += self.cfg.risk_extra_cost
            dmg_exp += self.cfg.risky_damage
            if self.cfg.p_ambush > 0.0:
                c += self.cfg.p_ambush * self.cfg.ambush_penalty
                dmg_exp += self.cfg.p_ambush * self.cfg.ambush_damage

        if self._tower_in_range(agent_next):
            c += self.cfg.tower_penalty
            dmg_exp += self.cfg.tower_damage

        if hp - dmg_exp <= 0:
            c += self.cfg.death_penalty

        return c

    def _expected_cost_with_enemy_hp(self, s, a: int) -> float:
        agent, enemy, hp = s
        agent_next = self._next_det_agent(agent, a)
        if self.cfg.enemy_chase:
            enemy_next = self.astar_next_step(enemy, agent_next)
        else:
            enemy_next = enemy

        c = self.cfg.step_cost
        dmg_exp = 0.0

        if agent_next in self.risky:
            c += self.cfg.risk_extra_cost
            dmg_exp += self.cfg.risky_damage
            if self.cfg.p_ambush > 0.0:
                c += self.cfg.p_ambush * self.cfg.ambush_penalty
                dmg_exp += self.cfg.p_ambush * self.cfg.ambush_damage

        if agent_next == enemy_next:
            c += self.cfg.catch_penalty
            dmg_exp += self.cfg.catch_damage

        if self._tower_in_range(agent_next):
            c += self.cfg.tower_penalty
            dmg_exp += self.cfg.tower_damage

        if hp - dmg_exp <= 0:
            c += self.cfg.death_penalty

        return c

    def cost(self, s: Any, a: int) -> float:
        """
        给 Value Iteration 用的“期望单步代价”。
        """
        if not self.use_enemy_state and not self.use_hp_state:
            return self._expected_cost_single(s, a)
        if self.use_enemy_state and not self.use_hp_state:
            return self._expected_cost_with_enemy(s, a)
        if not self.use_enemy_state and self.use_hp_state:
            return self._expected_cost_with_hp(s, a)
        return self._expected_cost_with_enemy_hp(s, a)

    # ------------------------------------------------------------------
    #                       transition（给 VI 用）
    # ------------------------------------------------------------------
    def transition(self, s: Any, a: int):
        """
        这里保持简化：所有转移都是“确定性”的，
        随机性都折进 cost() 的期望里。
        """
        # 模式 1：仅位置
        if not self.use_enemy_state and not self.use_hp_state:
            s2 = self._next_det_agent(s, a)
            return [1.0], [s2]

        # 模式 2：位置 + 敌人
        if self.use_enemy_state and not self.use_hp_state:
            agent, enemy = s
            agent_next = self._next_det_agent(agent, a)
            if self.cfg.enemy_chase:
                enemy_next = self.astar_next_step(enemy, agent_next)
            else:
                enemy_next = enemy
            return [1.0], [(agent_next, enemy_next)]

        # 模式 3：位置 + HP
        if not self.use_enemy_state and self.use_hp_state:
            agent, hp = s
            agent_next = self._next_det_agent(agent, a)

            dmg_exp = 0.0
            if agent_next in self.risky:
                dmg_exp += self.cfg.risky_damage
                if self.cfg.p_ambush > 0.0:
                    dmg_exp += self.cfg.p_ambush * self.cfg.ambush_damage
            if self._tower_in_range(agent_next):
                dmg_exp += self.cfg.tower_damage

            hp_next = max(0, int(round(hp - dmg_exp)))
            return [1.0], [(agent_next, hp_next)]

        # 模式 4：位置 + 敌人 + HP
        agent, enemy, hp = s
        agent_next = self._next_det_agent(agent, a)
        if self.cfg.enemy_chase:
            enemy_next = self.astar_next_step(enemy, agent_next)
        else:
            enemy_next = enemy

        dmg_exp = 0.0
        if agent_next in self.risky:
            dmg_exp += self.cfg.risky_damage
            if self.cfg.p_ambush > 0.0:
                dmg_exp += self.cfg.p_ambush * self.cfg.ambush_damage
        if agent_next == enemy_next:
            dmg_exp += self.cfg.catch_damage
        if self._tower_in_range(agent_next):
            dmg_exp += self.cfg.tower_damage

        hp_next = max(0, int(round(hp - dmg_exp)))
        return [1.0], [(agent_next, enemy_next, hp_next)]

    # ------------------------------------------------------------------
    #                       sample_step（给 QL 用）
    # ------------------------------------------------------------------
    def sample_step(self, s: Any, a: int):
        """
        给 Q-learning 用的真实交互：
          - 风险格随机埋伏
          - 塔真实扣血
          - 敌人真实追击
        """
        # 模式 1：仅位置
        if not self.use_enemy_state and not self.use_hp_state:
            s2 = self._next_det_agent(s, a)
            c = self.cfg.step_cost

            if s2 in self.risky:
                c += self.cfg.risk_extra_cost
                if self.cfg.p_ambush > 0.0 and self.cfg.ambush_penalty > 0.0:
                    if self._rng.rand() < self.cfg.p_ambush:
                        c += self.cfg.ambush_penalty

            if self._tower_in_range(s2):
                c += self.cfg.tower_penalty

            done = self.is_goal_single(s2)
            return s2, c, done

        # 模式 2：位置 + 敌人
        if self.use_enemy_state and not self.use_hp_state:
            agent, enemy = s
            agent_next = self._next_det_agent(agent, a)
            if self.cfg.enemy_chase:
                enemy_next = self.astar_next_step(enemy, agent_next)
            else:
                enemy_next = enemy

            c = self.cfg.step_cost

            if agent_next in self.risky:
                c += self.cfg.risk_extra_cost
                if self.cfg.p_ambush > 0.0 and self.cfg.ambush_penalty > 0.0:
                    if self._rng.rand() < self.cfg.p_ambush:
                        c += self.cfg.ambush_penalty

            done = False
            if agent_next == enemy_next:
                c += self.cfg.catch_penalty
                done = True
            else:
                if self.goal is not None and agent_next == self.goal:
                    done = True

            if self._tower_in_range(agent_next):
                c += self.cfg.tower_penalty

            return (agent_next, enemy_next), c, done

        # 模式 3：位置 + HP
        if not self.use_enemy_state and self.use_hp_state:
            agent, hp = s
            agent_next = self._next_det_agent(agent, a)

            c = self.cfg.step_cost
            hp2 = hp

            if agent_next in self.risky:
                c += self.cfg.risk_extra_cost
                hp2 -= self.cfg.risky_damage
                if self.cfg.p_ambush > 0.0 and self.cfg.ambush_penalty > 0.0:
                    if self._rng.rand() < self.cfg.p_ambush:
                        c += self.cfg.ambush_penalty
                        hp2 -= self.cfg.ambush_damage

            if self._tower_in_range(agent_next):
                c += self.cfg.tower_penalty
                hp2 -= self.cfg.tower_damage

            done = False
            if hp2 <= 0:
                c += self.cfg.death_penalty
                hp2 = 0
                done = True

            if self.goal is not None and agent_next == self.goal:
                done = True

            return (agent_next, hp2), c, done

        # 模式 4：位置 + 敌人 + HP
        agent, enemy, hp = s
        agent_next = self._next_det_agent(agent, a)
        if self.cfg.enemy_chase:
            enemy_next = self.astar_next_step(enemy, agent_next)
        else:
            enemy_next = enemy

        c = self.cfg.step_cost
        hp2 = hp

        if agent_next in self.risky:
            c += self.cfg.risk_extra_cost
            hp2 -= self.cfg.risky_damage
            if self.cfg.p_ambush > 0.0 and self.cfg.ambush_penalty > 0.0:
                if self._rng.rand() < self.cfg.p_ambush:
                    c += self.cfg.ambush_penalty
                    hp2 -= self.cfg.ambush_damage

        if agent_next == enemy_next:
            c += self.cfg.catch_penalty
            hp2 -= self.cfg.catch_damage

        if self._tower_in_range(agent_next):
            c += self.cfg.tower_penalty
            hp2 -= self.cfg.tower_damage

        done = False
        if hp2 <= 0:
            c += self.cfg.death_penalty
            hp2 = 0
            done = True

        if self.goal is not None and agent_next == self.goal:
            done = True

        return (agent_next, enemy_next, hp2), c, done

    #               A* 寻路（MOBA 对战 / 敌人追击用）
    def astar_next_step(self, start: Coord, goal: Coord) -> Coord:
        """
        A* 寻路：返回从 start 朝 goal 走的“下一步”。
        仍然以“避开风险格”为主，塔不参与 path cost（只参与战斗伤害）。
        """
        if start == goal:
            return start

        pq: List[Tuple[float, Coord]] = []
        heapq.heappush(pq, (0.0, start))

        came: Dict[Coord, Optional[Coord]] = {start: None}
        cost: Dict[Coord, float] = {start: 0.0}

        while pq:
            _, cur = heapq.heappop(pq)
            if cur == goal:
                break

            for d in ACTIONS.values():
                nxt = (cur[0] + d[0], cur[1] + d[1])
                if not self.in_bounds(nxt):
                    continue

                base = 1.0 + (self.cfg.risk_extra_cost if nxt in self.risky else 0.0)
                new_cost = cost[cur] + base

                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    priority = new_cost + abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                    heapq.heappush(pq, (priority, nxt))
                    came[nxt] = cur

        if goal not in came:
            return start

        step = goal
        while came[step] != start:
            step = came[step]
        return step
