# -*- coding: utf-8 -*-
"""
web/services/robot.py — SO-101（LeRobot）视觉引导机械臂执行层接口（阶段6.1 mock）
================================================================================
在 Agent 决策基础上，输出机械臂执行指令（任务 / 动作 / 目标区域），
模拟「病害定位与样本采集」等物理执行动作。

当前阶段（mock）：不连接真实机械臂、不安装 LeRobot，使用 MockRobotBackend
模拟执行并返回结果，验证接口与展示链路。

未来接入（预留，保持 execute_task 签名不变）：
    - Hugging Face LeRobot
    - SO-101 follower API
只需将 MockRobotBackend 替换为 LeRobotSO101Backend（实现 execute()），
Web 层与其余模块零改动。

本模块不依赖 Paddle、不依赖 inference 内部实现、不修改 PP-YOLOE 模型，
也不影响 severity.py / vlm.py / rag.py / agent.py / speech.py 现有流程。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class RobotBackend(ABC):
    """机械臂执行后端抽象接口：LeRobot / SO-101 follower 等统一入口。"""

    backend_name: str = "base"
    robot_name: str = "未连接"
    status: str = "未连接"

    @abstractmethod
    def execute(self, task, action, target) -> dict:
        """执行机械臂任务，返回执行结果。

        返回：
            {
                "status":  执行状态（模拟执行 / 跳过 / ...）,
                "robot":   设备名称,
                "action":  执行动作,
                "target":  目标区域,
                "message": 执行结果说明,
            }
        """


class MockRobotBackend(RobotBackend):
    """mock 机械臂：不连接真实硬件，模拟执行并返回结果。"""

    backend_name = "mock"
    robot_name = "SO-101（mock）"
    status = "mock模式（SO-101未连接）"

    def execute(self, task, action, target) -> dict:
        if not task or not action:
            return {
                "status": "跳过",
                "robot": self.robot_name,
                "action": action or "",
                "target": target or "",
                "message": "无待执行任务（可能尚未完成检测或未生成决策），机械臂保持待机。",
            }
        return {
            "status": "模拟执行",
            "robot": self.robot_name,
            "action": action,
            "target": target,
            "message": (
                f"已模拟执行「{task}」：机械臂将移动至目标区域 {target or '—'}，"
                "完成病害样本定位与采集动作（当前为 mock，未驱动真实 SO-101 硬件）。"
            ),
        }


class LeRobotSO101Backend(RobotBackend):
    """预留：Hugging Face LeRobot + SO-101 follower API 真实控制。"""

    backend_name = "lerobot_so101"
    robot_name = "SO-101"
    status = "LeRobot（预留）"

    def execute(self, task, action, target) -> dict:
        raise NotImplementedError(
            "LeRobotSO101Backend 尚未实现（预留接口）。"
            "需安装 LeRobot 并连接 SO-101 follower 后方可使用。"
        )


# 当前默认后端（mock）。未来接入真实机械臂时替换为 LeRobotSO101Backend 即可。
_default_backend: RobotBackend = MockRobotBackend()


def execute_task(task, action, target) -> dict:
    """对外统一入口：执行机械臂任务（当前 mock 模拟）。

    参数：
        task    Agent 生成的任务
        action  Agent 推荐动作
        target  Agent 目标区域

    返回：
        {
            "status":  执行状态,
            "robot":   设备名称,
            "action":  执行动作,
            "target":  目标区域,
            "message": 执行结果说明,
        }
    """
    return _default_backend.execute(task, action, target)


def get_robot_status() -> str:
    """返回当前机械臂后端状态（如 'mock模式（SO-101未连接）'）。"""
    return _default_backend.status
