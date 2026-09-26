# -*- coding: utf-8 -*-
"""
NN-VMC Workbench v2.2 (Optimized) - 神经网络变分蒙特卡洛桌面工作台
==========================================================
基于 JaQMC (JAX) 框架的图形化 NN-QMC 计算平台

v2.1 改进（借鉴东南·云霄设计理念）：
  ✅ 三栏布局（分子库 | 3D可视化+参数 | YAML实时预览）
  ✅ 三档优化级别大按钮（快速/平衡/高性能）
  ✅ 分子库彩色分类（按化学键类型）
  ✅ YAML配置双向同步（可直接编辑代码）
  ✅ 新手引导（首次打开使用指南）
  ✅ 右键上下文菜单
  ✅ 专业深色/浅色主题切换
  ✅ 16分子基准面板一键加载

功能：
  1. 分子构建（预设/自定义 + 3D可视化）
  2. 智能配置推荐（Forward Laplacian / sparse / 赝势）
  3. 计算管理（提交/监控/日志）
  4. 批量计算队列（多任务顺序执行/状态管理）
  5. 结果分析（收敛曲线/能量统计 + 论文图表一键导出）

依赖：Python 3.12+, jaqmc, pyscf, matplotlib, numpy, h5py
运行：python nnvmc_workbench.py
"""

import os
import sys
import queue
import time
import threading
import subprocess
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np

# matplotlib 嵌入 Tk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# 主题配色
# ============================================================
THEMES = {
    "light": {
        "bg": "#f8fafc", "fg": "#1e293b", "accent": "#3b82f6",
        "accent_light": "#dbeafe", "card": "#ffffff", "border": "#e2e8f0",
        "hover": "#f1f5f9", "text_bg": "#ffffff", "status": "#10b981",
        "warning": "#f59e0b", "danger": "#ef4444", "text_muted": "#64748b",
    },
    "dark": {
        "bg": "#0f172a", "fg": "#e2e8f0", "accent": "#60a5fa",
        "accent_light": "#1e3a5f", "card": "#1e293b", "border": "#334155",
        "hover": "#334155", "text_bg": "#0f172a", "status": "#34d399",
        "warning": "#fbbf24", "danger": "#f87171", "text_muted": "#94a3b8",
    },
}

# 分子分类颜色（借鉴东南·云霄量子门彩色分类）
MOLECULE_CATEGORY_COLORS = {
    "双原子": "#f87171",      # 柔和红
    "线性": "#60a5fa",        # 柔和蓝
    "平面": "#34d399",        # 柔和绿
    "四面体": "#fbbf24",      # 柔和黄
    "三角锥": "#a78bfa",      # 柔和紫
    "芳香": "#fb923c",        # 柔和橙
    "PH赝势": "#22d3ee",      # 柔和青
}


# ============================================================
# v2.2 新增：9分子生产计算基准数据
# ============================================================
BENCHMARK_RESULTS = {
    "HF (氟化氢, 10e)": {
        "n_electrons": 10,
        "nnvmc_energy": -100.481,
        "ccsdt_reference": -100.459,
        "absolute_error_mha": 21.82,
        "error_kcal_mol": 13.69,
        "status": "completed",
        "note": "中等精度",
    },
    "CH4 (甲烷, 10e)": {
        "n_electrons": 10,
        "nnvmc_energy": -40.515,
        "ccsdt_reference": -40.518,
        "absolute_error_mha": 3.17,
        "error_kcal_mol": 1.99,
        "status": "completed",
        "note": "接近化学精度",
    },
    "N2 (氮气, 14e)": {
        "n_electrons": 14,
        "nnvmc_energy": -109.529,
        "ccsdt_reference": -109.538,
        "absolute_error_mha": 9.44,
        "error_kcal_mol": 5.92,
        "status": "completed",
        "note": "良好精度",
    },
    "CO (一氧化碳, 14e)": {
        "n_electrons": 14,
        "nnvmc_energy": -113.352,
        "ccsdt_reference": -113.339,
        "absolute_error_mha": 12.84,
        "error_kcal_mol": 8.06,
        "status": "completed",
        "note": "良好精度",
    },
    "C2H2 (乙炔, 14e)": {
        "n_electrons": 14,
        "nnvmc_energy": -77.335,
        "ccsdt_reference": -77.313,
        "absolute_error_mha": 22.39,
        "error_kcal_mol": 14.05,
        "status": "completed",
        "note": "中等精度",
    },
    "NH3 (氨, 10e)": {
        "n_electrons": 10,
        "nnvmc_energy": -56.536,
        "ccsdt_reference": -56.558,
        "absolute_error_mha": 21.62,
        "error_kcal_mol": 13.57,
        "status": "completed",
        "note": "中等精度",
    },
    "C2H4 (乙烯, 16e)": {
        "n_electrons": 16,
        "nnvmc_energy": -78.568,
        "ccsdt_reference": -78.524,
        "absolute_error_mha": 43.92,
        "error_kcal_mol": 27.56,
        "status": "completed",
        "note": "中等精度",
    },
    "H2O (水, 10e)": {
        "n_electrons": 10,
        "nnvmc_energy": -76.439,
        "ccsdt_reference": -76.480,
        "absolute_error_mha": 41.04,
        "error_kcal_mol": 25.75,
        "status": "completed",
        "note": "中等精度",
    },
    "HCHO (甲醛, 16e)": {
        "n_electrons": 16,
        "nnvmc_energy": -114.363,
        "ccsdt_reference": -114.418,
        "absolute_error_mha": 55.12,
        "error_kcal_mol": 34.58,
        "status": "completed",
        "note": "较大误差",
    },
}

# 精度统计
BENCHMARK_STATS = {
    "mean_error_mha": 25.7,
    "median_error_mha": 21.8,
    "min_error_mha": 3.17,
    "max_error_mha": 55.12,
    "chemical_accuracy_count": 0,
    "near_chemical_accuracy_count": 1,
    "total_completed": 9,
    "total_panel": 16,
}


# ============================================================
# 预设分子数据库（坐标单位：bohr）
# ============================================================
PRESET_MOLECULES = {
    # --- 第一篇6分子 ---
    "H2O (水, 10e)": {
        "atoms": [
            {"symbol": "O", "coords": [0.0, 0.0, 0.0]},
            {"symbol": "H", "coords": [1.107157, 1.430429, 0.0]},
            {"symbol": "H", "coords": [-1.107157, 1.430429, 0.0]},
        ],
        "n_elec": 10, "category": "平面", "pp_supported": {"AE": True, "ECP": True, "PH": False},
    },
    "H2S (硫化氢, 18e)": {
        "atoms": [
            {"symbol": "S", "coords": [0.0, 0.0, 0.0]},
            {"symbol": "H", "coords": [1.751359, 1.817390, 0.0]},
            {"symbol": "H", "coords": [-1.751359, 1.817390, 0.0]},
        ],
        "n_elec": 18, "category": "平面", "pp_supported": {"AE": True, "ECP": True, "PH": True},
    },
    "N2 (氮气, 14e)": {
        "atoms": [
            {"symbol": "N", "coords": [0.0, 0.0, 1.038]},
            {"symbol": "N", "coords": [0.0, 0.0, -1.038]},
        ],
        "n_elec": 14, "category": "双原子", "pp_supported": {"AE": True, "ECP": True, "PH": False},
    },
    "CO2 (二氧化碳, 22e)": {
        "atoms": [
            {"symbol": "C", "coords": [0.0, 0.0, 0.0]},
            {"symbol": "O", "coords": [0.0, 0.0, 2.196]},
            {"symbol": "O", "coords": [0.0, 0.0, -2.196]},
        ],
        "n_elec": 22, "category": "线性", "pp_supported": {"AE": True, "ECP": True, "PH": False},
    },
    "CH4 (甲烷, 10e)": {
        "atoms": [
            {"symbol": "C", "coords": [0.0, 0.0, 0.0]},
            {"symbol": "H", "coords": [1.184, 1.184, 1.184]},
            {"symbol": "H", "coords": [-1.184, -1.184, 1.184]},
            {"symbol": "H", "coords": [-1.184, 1.184, -1.184]},
            {"symbol": "H", "coords": [1.184, -1.184, -1.184]},
        ],
        "n_elec": 10, "category": "四面体", "pp_supported": {"AE": True, "ECP": True, "PH": False},
    },
    "NH3 (氨, 10e)": {
        "atoms": [
            {"symbol": "N", "coords": [0.0, 0.0, 0.0]},
            {"symbol": "H", "coords": [1.773, 0.0, -1.126]},
            {"symbol": "H", "coords": [-0.886, 1.535, -1.126]},
            {"symbol": "H", "coords": [-0.886, -1.535, -1.126]},
        ],
        "n_elec": 10, "category": "三角锥", "pp_supported": {"AE": True, "ECP": True, "PH": False},
    },
    # --- 第二篇10个新分子 ---
    "HF (氟化氢, 10e)": {
        "atoms": [
            {"symbol": "H", "coords": [0.0, 0.0, 0.0]},
            {"symbol": "F", "coords": [0.0, 0.0, 1.732]},
        ],
        "n_elec": 10, "category": "双原子", "pp_supported": {"AE": True, "ECP": True, "PH": False},
    },
    "CO (一氧化碳, 14e)": {
        "atoms": [
            {"symbol": "C", "coords": [0.0, 0.0, 0.0]},
            {"symbol": "O", "coords": [0.0, 0.0, 2.132]},
        ],
        "n_elec": 14, "category": "双原子", "pp_supported": {"AE": True, "ECP": True, "PH": False},
    },
    "C2H2 (乙炔, 14e)": {
        "atoms": [
            {"symbol": "C", "coords": [0.0, 0.0, 1.061]},
            {"symbol": "C", "coords": [0.0, 0.0, -1.061]},
            {"symbol": "H", "coords": [0.0, 0.0, 2.668]},
            {"symbol": "H", "coords": [0.0, 0.0, -2.668]},
        ],
        "n_elec": 14, "category": "线性", "pp_supported": {"AE": True, "ECP": True, "PH": False},
    },
    "C2H4 (乙烯, 16e)": {
        "atoms": [
            {"symbol": "C", "coords": [0.0, 0.0, 1.261]},
            {"symbol": "C", "coords": [0.0, 0.0, -1.261]},
            {"symbol": "H", "coords": [0.0, 1.743, 2.338]},
            {"symbol": "H", "coords": [0.0, -1.743, 2.338]},
            {"symbol": "H", "coords": [0.0, 1.743, -2.338]},
            {"symbol": "H", "coords": [0.0, -1.743, -2.338]},
        ],
        "n_elec": 16, "category": "平面", "pp_supported": {"AE": True, "ECP": True, "PH": False},
    },
    "HCHO (甲醛, 16e)": {
        "atoms": [
            {"symbol": "C", "coords": [0.0, 0.0, 0.0]},
            {"symbol": "O", "coords": [0.0, 0.0, 2.273]},
            {"symbol": "H", "coords": [1.894, 0.0, -1.102]},
            {"symbol": "H", "coords": [-1.894, 0.0, -1.102]},
        ],
        "n_elec": 16, "category": "平面", "pp_supported": {"AE": True, "ECP": True, "PH": False},
    },
    "CH3OH (甲醇, 18e)": {
        "atoms": [
            {"symbol": "C", "coords": [0.0, 0.0, 0.0]},
            {"symbol": "O", "coords": [0.0, 0.0, 2.720]},
            {"symbol": "H", "coords": [0.0, 0.0, 3.684]},
            {"symbol": "H", "coords": [1.888, 0.0, -1.085]},
            {"symbol": "H", "coords": [-0.944, 1.635, -1.085]},
            {"symbol": "H", "coords": [-0.944, -1.635, -1.085]},
        ],
        "n_elec": 18, "category": "四面体", "pp_supported": {"AE": True, "ECP": True, "PH": False},
    },
    "PH3 (磷化氢, 8价e, PH)": {
        "atoms": [
            {"symbol": "P", "coords": [0.0, 0.0, 0.0]},
            {"symbol": "H", "coords": [2.396, 0.0, -1.374]},
            {"symbol": "H", "coords": [-1.198, 2.075, -1.374]},
            {"symbol": "H", "coords": [-1.198, -2.075, -1.374]},
        ],
        "n_elec": 8, "category": "PH赝势", "pp_supported": {"AE": False, "ECP": True, "PH": True},
    },
    "HCl (氯化氢, 8价e, PH)": {
        "atoms": [
            {"symbol": "H", "coords": [0.0, 0.0, 0.0]},
            {"symbol": "Cl", "coords": [0.0, 0.0, 2.409]},
        ],
        "n_elec": 8, "category": "PH赝势", "pp_supported": {"AE": False, "ECP": True, "PH": True},
    },
    "SO2 (二氧化硫, 18价e, PH)": {
        "atoms": [
            {"symbol": "S", "coords": [0.0, 0.0, 0.0]},
            {"symbol": "O", "coords": [2.196, 0.0, 1.430]},
            {"symbol": "O", "coords": [-2.196, 0.0, 1.430]},
        ],
        "n_elec": 18, "category": "PH赝势", "pp_supported": {"AE": False, "ECP": True, "PH": True},
    },
    "C6H6 (苯, 42e)": {
        "atoms": [
            {"symbol": "C", "coords": [2.638, 0.0, 0.0]},
            {"symbol": "C", "coords": [1.319, 2.285, 0.0]},
            {"symbol": "C", "coords": [-1.319, 2.285, 0.0]},
            {"symbol": "C", "coords": [-2.638, 0.0, 0.0]},
            {"symbol": "C", "coords": [-1.319, -2.285, 0.0]},
            {"symbol": "C", "coords": [1.319, -2.285, 0.0]},
            {"symbol": "H", "coords": [4.679, 0.0, 0.0]},
            {"symbol": "H", "coords": [2.339, 4.052, 0.0]},
            {"symbol": "H", "coords": [-2.339, 4.052, 0.0]},
            {"symbol": "H", "coords": [-4.679, 0.0, 0.0]},
            {"symbol": "H", "coords": [-2.339, -4.052, 0.0]},
            {"symbol": "H", "coords": [2.339, -4.052, 0.0]},
        ],
        "n_elec": 42, "category": "芳香", "pp_supported": {"AE": True, "ECP": True, "PH": False},
    },
}

# 原子颜色 (CPK配色) 和显示半径
ATOM_COLORS = {
    "H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F",
    "Br": "#A62929", "I": "#940094", "He": "#D9FFFF", "Ne": "#B3E3F5",
    "Ar": "#83D0E5", "Kr": "#5CB8D1", "Xe": "#429EB0",
}
ATOM_RADII = {
    "H": 100, "C": 200, "N": 180, "O": 170, "F": 150,
    "P": 250, "S": 240, "Cl": 220, "Br": 240, "I": 260,
}
BOND_THRESHOLD = 3.0  # bohr，小于此距离认为成键

HA_TO_KCAL = 627.5094740631


# ============================================================
# 三档优化级别配置（借鉴东南·云霄优化级别一/二/三）
# ============================================================
OPTIMIZATION_LEVELS = {
    "fast": {
        "name": "⚡ 快速",
        "desc": "冒烟测试\n3步训练\n调试用",
        "batch_size": 256, "ndets": 2,
        "hidden_single": [32, 32, 32], "hidden_double": [4, 4, 4],
        "pretrain_steps": 10, "train_steps": 3,
        "color": "#22c55e",
    },
    "balanced": {
        "name": "⚖️ 平衡",
        "desc": "中等规模\n5000步训练\n初步结果",
        "batch_size": 2048, "ndets": 8,
        "hidden_single": [128, 128, 128], "hidden_double": [16, 16, 16],
        "pretrain_steps": 500, "train_steps": 5000,
        "color": "#3b82f6",
    },
    "high": {
        "name": "🚀 高性能",
        "desc": "生产计算\n20000步训练\n论文级精度",
        "batch_size": 4096, "ndets": 16,
        "hidden_single": [256, 256, 256], "hidden_double": [32, 32, 32],
        "pretrain_steps": 1000, "train_steps": 20000,
        "color": "#f97316",
    },
}


# ============================================================
# 配置生成器
# ============================================================
def generate_yaml_config(mol_name, atoms, pp_type, wf_arch, ndets,
                          hidden_single, hidden_double, batch_size,
                          pretrain_steps, train_steps, use_fl, use_sparse,
                          save_path, seed=42):
    lines = []
    lines.append("# JaQMC configuration - generated by NN-VMC Workbench v2.1")
    lines.append(f"# Molecule: {mol_name}")
    lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("workflow:")
    lines.append(f"  seed: {seed}")
    lines.append(f"  batch_size: {batch_size}")
    lines.append(f"  save_path: {save_path}")
    lines.append("")
    lines.append("system:")
    lines.append("  unit: bohr")
    lines.append("  atoms:")
    for atom in atoms:
        c = atom["coords"]
        lines.append(f"    - symbol: {atom['symbol']}")
        lines.append(f"      coords: [{c[0]}, {c[1]}, {c[2]}]")
    if pp_type != "AE":
        lines.append("  pp:")
        pp_val = "ccecp" if pp_type == "ECP" else "ph"
        symbols = list(set(a["symbol"] for a in atoms))
        for s in symbols:
            lines.append(f"    {s}: {pp_val}")
    lines.append("")
    lines.append("wf:")
    lines.append(f"  architecture: {wf_arch.lower()}")
    lines.append(f"  ndets: {ndets}")
    lines.append(f"  hidden_dims_single: {hidden_single}")
    lines.append(f"  hidden_dims_double: {hidden_double}")
    lines.append(f"  use_forward_laplacian: {str(use_fl).lower()}")
    if use_fl:
        lines.append(f"  use_sparse_derivatives: {str(use_sparse).lower()}")
    lines.append("")
    lines.append("pretrain:")
    lines.append("  run:")
    lines.append(f"    iterations: {pretrain_steps}")
    lines.append("")
    lines.append("train:")
    lines.append("  run:")
    lines.append(f"    iterations: {train_steps}")
    lines.append("  optimizer: kfac")
    lines.append("")
    return "\n".join(lines)


def recommend_config(n_elec, has_ph_support=False):
    rec = {}
    rec["use_fl"] = True
    rec["use_sparse"] = n_elec > 15
    if n_elec <= 10:
        rec["ndets"] = 4
        rec["hidden_single"] = [64, 64, 64]
        rec["hidden_double"] = [8, 8, 8]
        rec["batch_size"] = 512
    elif n_elec <= 18:
        rec["ndets"] = 8
        rec["hidden_single"] = [128, 128, 128]
        rec["hidden_double"] = [16, 16, 16]
        rec["batch_size"] = 1024
    else:
        rec["ndets"] = 16
        rec["hidden_single"] = [256, 256, 256]
        rec["hidden_double"] = [32, 32, 32]
        rec["batch_size"] = 2048
    rec["pp"] = "PH" if has_ph_support else "AE"
    speedup = {10: 2.0, 14: 2.0, 18: 4.0, 22: 2.8}.get(n_elec, 2.5)
    rec["estimated_speedup"] = speedup
    return rec


# ============================================================
# 计算进程管理（单个任务）
# ============================================================
class CalculationManager:
    def __init__(self, log_callback, status_callback, finish_callback=None):
        self.process = None
        self.log_callback = log_callback
        self.status_callback = status_callback
        self.finish_callback = finish_callback
        self.running = False
        self.output_dir = None

    def start(self, yaml_path, jaqmc_path, conda_lib_path):
        if self.running:
            return False
        self.running = True
        self.output_dir = str(Path(yaml_path).parent)

        def run():
            env = os.environ.copy()
            env["PATH"] = conda_lib_path + os.pathsep + env.get("PATH", "")
            try:
                self.process = subprocess.Popen(
                    [jaqmc_path, "molecule", "train", "--yml", yaml_path],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, env=env, bufsize=1,
                )
                self.status_callback("运行中...")
                for line in self.process.stdout:
                    self.log_callback(line.rstrip())
                self.process.wait()
                rc = self.process.returncode
                self.status_callback(f"完成 (exit {rc})")
                if self.finish_callback:
                    self.finish_callback(rc)
            except Exception as e:
                self.log_callback(f"[ERROR] {e}")
                self.status_callback("错误")
                if self.finish_callback:
                    self.finish_callback(-1)
            finally:
                self.running = False
                self.process = None

        threading.Thread(target=run, daemon=True).start()
        return True

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.status_callback("已停止")
            self.running = False
            return True
        return False


# ============================================================
# 批量计算管理
# ============================================================
class BatchManager:
    def __init__(self, log_callback, status_callback, list_update_callback):
        self.queue = queue.Queue()
        self.tasks = []  # [{"name", "yaml_path", "status", "result"}]
        self.log_callback = log_callback
        self.status_callback = status_callback
        self.list_update_callback = list_update_callback
        self.running = False
        self.current_task = None
        self.process = None

    def add_task(self, name, yaml_path):
        task = {"name": name, "yaml_path": yaml_path, "status": "等待", "result": ""}
        self.tasks.append(task)
        self.queue.put(task)
        self.list_update_callback()

    def remove_task(self, index):
        if 0 <= index < len(self.tasks):
            if self.tasks[index]["status"] == "等待":
                self.tasks.pop(index)
                self.list_update_callback()

    def clear_completed(self):
        self.tasks = [t for t in self.tasks if t["status"] in ("等待", "运行中")]
        self.list_update_callback()

    def start(self, jaqmc_path, conda_lib_path):
        if self.running:
            return
        self.running = True

        def run():
            while self.running and not self.queue.empty():
                task = self.queue.get()
                if task["status"] != "等待":
                    continue
                self.current_task = task
                task["status"] = "运行中"
                self.list_update_callback()
                self.status_callback(f"运行: {task['name']}")
                self.log_callback(f"\n{'='*60}\n开始计算: {task['name']}\n{'='*60}\n")

                env = os.environ.copy()
                env["PATH"] = conda_lib_path + os.pathsep + env.get("PATH", "")
                try:
                    self.process = subprocess.Popen(
                        [jaqmc_path, "molecule", "train", "--yml", task["yaml_path"]],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, env=env, bufsize=1,
                    )
                    for line in self.process.stdout:
                        self.log_callback(line.rstrip())
                    self.process.wait()
                    rc = self.process.returncode
                    task["status"] = "完成" if rc == 0 else f"失败({rc})"
                    task["result"] = f"exit {rc}"
                except Exception as e:
                    task["status"] = "错误"
                    task["result"] = str(e)
                    self.log_callback(f"[ERROR] {e}")
                self.list_update_callback()

            self.running = False
            self.current_task = None
            self.status_callback("批量计算完成")

        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self.running = False
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.status_callback("已停止")


# ============================================================
# 新手引导窗口
# ============================================================
class OnboardingWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("欢迎使用 NN-VMC Workbench v2.1")
        self.geometry("720x560")
        self.configure(bg="#f5f5f5")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        # 标题
        title_frame = tk.Frame(self, bg="#2563eb", height=80)
        title_frame.pack(fill="x")
        tk.Label(title_frame, text="🧪 NN-VMC Workbench v2.1",
                 font=("Segoe UI", 20, "bold"), fg="white", bg="#2563eb").pack(pady=20)

        # 内容
        content = tk.Frame(self, bg="#f5f5f5")
        content.pack(fill="both", expand=True, padx=30, pady=20)

        steps = [
            ("1️⃣", "选择分子", "从左侧分子库选择预设分子，或自定义输入坐标。16个基准分子已内置，按化学键类型彩色分类。"),
            ("2️⃣", "配置参数", "选择优化级别（快速/平衡/高性能），或手动调整参数。右侧实时预览YAML配置，支持双向编辑。"),
            ("3️⃣", "运行计算", "点击开始计算，实时监控进度和日志。支持批量队列，多任务顺序执行。"),
            ("4️⃣", "分析结果", "查看收敛曲线、能量统计，一键导出论文级图表。"),
        ]

        for icon, title, desc in steps:
            row = tk.Frame(content, bg="#f5f5f5")
            row.pack(fill="x", pady=8)
            tk.Label(row, text=icon, font=("Segoe UI", 24), bg="#f5f5f5").pack(side="left", padx=10)
            text_frame = tk.Frame(row, bg="#f5f5f5")
            text_frame.pack(side="left", fill="x", expand=True)
            tk.Label(text_frame, text=title, font=("Segoe UI", 12, "bold"),
                     bg="#f5f5f5", fg="#202020", anchor="w").pack(fill="x")
            tk.Label(text_frame, text=desc, font=("Segoe UI", 10),
                     bg="#f5f5f5", fg="#505050", anchor="w", wraplength=520, justify="left").pack(fill="x")

        # 底部按钮
        btn_frame = tk.Frame(self, bg="#f5f5f5")
        btn_frame.pack(fill="x", pady=15)
        tk.Button(btn_frame, text="开始使用", font=("Segoe UI", 12, "bold"),
                  bg="#2563eb", fg="white", relief="flat", padx=30, pady=8,
                  command=self.destroy).pack()

        # 不再显示复选框
        self.show_again = tk.BooleanVar(value=True)
        tk.Checkbutton(btn_frame, text="启动时显示", variable=self.show_again,
                       bg="#f5f5f5", font=("Segoe UI", 9)).pack(pady=5)


# ============================================================
# 主应用类
# ============================================================
class NNVMWorkbench(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NN-VMC Workbench v2.1 — 神经网络变分蒙特卡洛工作台")
        self.geometry("1400x900")
        self.minsize(1200, 750)
        self.configure(bg=THEMES["light"]["bg"])

        # 状态变量
        self.current_molecule = None
        self.current_atoms = []
        self.current_yaml = ""
        self.opt_level = tk.StringVar(value="balanced")
        self.theme = "light"
        self.calc_manager = CalculationManager(self._log, self._status, self._calc_finished)
        self.batch_manager = BatchManager(self._batch_log, self._status, self._update_batch_list)

        # 路径配置
        self.jaqmc_path = tk.StringVar(value="C:\\jaqmc-conda\\Scripts\\jaqmc.exe")
        self.conda_lib_path = tk.StringVar(value="C:\\jaqmc-conda\\Library\\bin")
        self.work_dir = tk.StringVar(value=str(Path.cwd()))

        # 构建界面
        self._build_menu()
        self._build_toolbar()
        self._build_notebook()
        self._build_statusbar()

        # 显示新手引导
        self.after(500, self._show_onboarding)

    def _build_menu(self):
        menubar = tk.Menu(self)
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开配置...", command=self._open_config)
        file_menu.add_command(label="保存配置...", command=self._save_config)
        file_menu.add_separator()
        file_menu.add_command(label="导出结果...", command=self._export_results)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        # 工具菜单
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="环境设置...", command=self._env_settings)
        tools_menu.add_command(label="加载16分子基准面板", command=self._load_benchmark_panel)
        tools_menu.add_separator()
        tools_menu.add_command(label="切换主题", command=self._toggle_theme)
        menubar.add_cascade(label="工具", menu=tools_menu)
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="使用指南", command=self._show_onboarding)
        help_menu.add_command(label="关于", command=self._about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        self.config(menu=menubar)

    def _build_toolbar(self):
        toolbar = tk.Frame(self, bg="#ffffff", height=45, relief="raised", bd=1)
        toolbar.pack(fill="x", side="top")
        toolbar.pack_propagate(False)

        # Logo
        tk.Label(toolbar, text="🧪", font=("Segoe UI", 18), bg="#ffffff").pack(side="left", padx=10)
        tk.Label(toolbar, text="NN-VMC Workbench", font=("Segoe UI", 13, "bold"),
                 bg="#ffffff", fg="#2563eb").pack(side="left")
        tk.Label(toolbar, text="v2.1", font=("Segoe UI", 9),
                 bg="#ffffff", fg="#888888").pack(side="left", padx=5)

        tk.Frame(toolbar, width=2, bg="#e0e0e0").pack(side="left", fill="y", padx=15)

        # 快捷按钮
        buttons = [
            ("📂 打开", self._open_config),
            ("💾 保存", self._save_config),
            ("▶️ 运行", self._run_calculation),
            ("⏹️ 停止", self._stop_calculation),
            ("📊 分析", self._goto_results),
        ]
        for text, cmd in buttons:
            btn = tk.Button(toolbar, text=text, font=("Segoe UI", 10),
                            bg="#f5f5f5", fg="#333", relief="flat", padx=12, pady=5,
                            activebackground="#e0e0e0", command=cmd)
            btn.pack(side="left", padx=3)

        # 右侧：工作目录
        tk.Frame(toolbar, width=2, bg="#e0e0e0").pack(side="right", fill="y", padx=15)
        tk.Label(toolbar, text="📁", font=("Segoe UI", 12), bg="#ffffff").pack(side="right", padx=5)
        tk.Label(toolbar, textvariable=self.work_dir, font=("Consolas", 9),
                 bg="#ffffff", fg="#666").pack(side="right")


    def _build_benchmark_tab(self):
        """v2.2 新增：基准结果库标签页"""
        frame = tk.Frame(self.notebook, bg=THEMES["light"]["bg"])
        self.notebook.add(frame, text="  📈 基准结果库  ")
        
        # 顶部统计栏
        stats_frame = tk.Frame(frame, bg=THEMES["light"]["card"], relief="solid", bd=1)
        stats_frame.pack(fill="x", padx=10, pady=10)
        
        stats_title = tk.Label(stats_frame, text="📊 9分子生产计算基准结果统计", 
                              font=("Segoe UI", 14, "bold"), bg=THEMES["light"]["card"], fg=THEMES["light"]["fg"])
        stats_title.pack(pady=10)
        
        # 统计数字网格
        stats_grid = tk.Frame(stats_frame, bg=THEMES["light"]["card"])
        stats_grid.pack(pady=10)
        
        stats_items = [
            ("平均误差", f"{BENCHMARK_STATS['mean_error_mha']:.1f} mHa", "#3b82f6"),
            ("中位数误差", f"{BENCHMARK_STATS['median_error_mha']:.1f} mHa", "#22c55e"),
            ("最小误差", f"{BENCHMARK_STATS['min_error_mha']:.2f} mHa", "#16a34a"),
            ("最大误差", f"{BENCHMARK_STATS['max_error_mha']:.1f} mHa", "#ef4444"),
            ("化学精度达成", f"{BENCHMARK_STATS['chemical_accuracy_count']}/{BENCHMARK_STATS['total_completed']}", "#f59e0b"),
        ]
        
        for i, (label, value, color) in enumerate(stats_items):
            item_frame = tk.Frame(stats_grid, bg=THEMES["light"]["card"])
            item_frame.grid(row=0, column=i, padx=20)
            
            tk.Label(item_frame, text=label, font=("Segoe UI", 10), 
                    bg=THEMES["light"]["card"], fg="#666666").pack()
            tk.Label(item_frame, text=value, font=("Segoe UI", 16, "bold"), 
                    bg=THEMES["light"]["card"], fg=color).pack()
        
        # 主内容区：左侧表格 + 右侧图表
        main_frame = tk.Frame(frame, bg=THEMES["light"]["bg"])
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 左侧：分子列表表格
        left_frame = tk.Frame(main_frame, bg=THEMES["light"]["card"], relief="solid", bd=1)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        tk.Label(left_frame, text="分子精度详情", font=("Segoe UI", 12, "bold"), 
                bg=THEMES["light"]["card"], pady=10).pack()
        
        # Treeview表格
        columns = ("分子", "电子数", "误差(mHa)", "误差(kcal/mol)", "评价")
        tree = ttk.Treeview(left_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor="center")
        
        # 填充数据
        for mol_name, data in BENCHMARK_RESULTS.items():
            short_name = mol_name.split(" (")[0]
            tree.insert("", "end", values=(
                short_name,
                data["n_electrons"],
                f"{data['absolute_error_mha']:.2f}",
                f"{data['error_kcal_mol']:.2f}",
                data["note"],
            ))
        
        tree.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 右侧：误差柱状图
        right_frame = tk.Frame(main_frame, bg=THEMES["light"]["card"], relief="solid", bd=1)
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        tk.Label(right_frame, text="精度对比图", font=("Segoe UI", 12, "bold"), 
                bg=THEMES["light"]["card"], pady=10).pack()
        
        # matplotlib图
        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        mols = list(BENCHMARK_RESULTS.keys())
        errors = [BENCHMARK_RESULTS[m]["absolute_error_mha"] for m in mols]
        short_names = [m.split(" (")[0] for m in mols]
        
        # 按误差排序
        sorted_indices = np.argsort(errors)
        sorted_names = [short_names[i] for i in sorted_indices]
        sorted_errors = [errors[i] for i in sorted_indices]
        
        # 颜色：绿色<2, 蓝色<20, 红色>20
        colors = ["#22c55e" if e < 2 else "#3b82f6" if e < 20 else "#ef4444" for e in sorted_errors]
        
        ax.barh(sorted_names, sorted_errors, color=colors, edgecolor="black")
        ax.axvline(x=1.6, color="red", linestyle="--", linewidth=2, label="化学精度")
        ax.set_xlabel("绝对误差 (mHa)", fontsize=10)
        ax.set_title("NN-VMC + Forward Laplacian 精度基准", fontsize=11, fontweight="bold")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # 底部说明
        note_frame = tk.Frame(frame, bg=THEMES["light"]["card"], relief="solid", bd=1)
        note_frame.pack(fill="x", padx=10, pady=10)
        
        note_text = """
        📌 说明：
        • 计算配置：10000训练步，batch_size=4096，FermiNet 16行列式
        • GPU：RTX PRO 6000 98GB
        • 参考值：CCSD(T) 高精度量子化学计算
        • 化学精度：1.6 mHa (1 kcal/mol)
        """
        tk.Label(note_frame, text=note_text, font=("Segoe UI", 9), 
                bg=THEMES["light"]["card"], fg="#666666", justify="left", padx=10, pady=10).pack()

    def _build_notebook(self):
        style = ttk.Style()
        style.configure("TNotebook", background=self["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=[20, 10], font=("Segoe UI", 10))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        self._build_tab_molecule()
        self._build_tab_config()
        self._build_tab_calculation()
        self._build_tab_batch()
        self._build_tab_results()

    def _build_statusbar(self):
        self.statusbar = tk.Frame(self, bg="#2563eb", height=28)
        self.statusbar.pack(fill="x", side="bottom")
        self.statusbar.pack_propagate(False)

        self.status_var = tk.StringVar(value="就绪")
        tk.Label(self.statusbar, textvariable=self.status_var, font=("Segoe UI", 9),
                 bg="#2563eb", fg="white").pack(side="left", padx=10)

        tk.Label(self.statusbar, text="● 已连接 JaQMC", font=("Segoe UI", 9),
                 bg="#2563eb", fg="#a6e3a1").pack(side="right", padx=10)

    # ============================================================
    # 标签页1：分子构建（三栏布局）
    # ============================================================
    def _build_tab_molecule(self):
        frame = tk.Frame(self.notebook, bg=THEMES["light"]["bg"])
        self.notebook.add(frame, text="  🔬 分子构建  ")

        # 左栏：分子库（彩色分类）
        left = tk.Frame(frame, bg="#ffffff", width=320, relief="raised", bd=1)
        left.pack(side="left", fill="y", padx=5, pady=5)
        left.pack_propagate(False)

        tk.Label(left, text="📚 分子库", font=("Segoe UI", 12, "bold"),
                 bg="#ffffff", fg="#2563eb").pack(pady=10)

        # 分类筛选
        filter_frame = tk.Frame(left, bg="#ffffff")
        filter_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(filter_frame, text="筛选:", font=("Segoe UI", 9), bg="#ffffff").pack(side="left")
        self.mol_filter = tk.StringVar(value="全部")
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.mol_filter,
                                      values=["全部", "双原子", "线性", "平面", "四面体", "三角锥", "芳香", "PH赝势"],
                                      state="readonly", width=12)
        filter_combo.pack(side="left", padx=5)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_molecule_list())

        # 分子列表（带彩色标签）
        list_frame = tk.Frame(left, bg="#ffffff")
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.mol_listbox = tk.Listbox(list_frame, font=("Segoe UI", 10),
                                        bg="#fafafa", fg="#333", selectbackground="#2563eb",
                                        selectforeground="white", relief="flat", bd=0,
                                        highlightthickness=1, highlightcolor="#2563eb")
        self.mol_listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.mol_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.mol_listbox.config(yscrollcommand=scrollbar.set)
        self.mol_listbox.bind("<<ListboxSelect>>", self._on_molecule_select)
        self.mol_listbox.bind("<Double-Button-1>", lambda e: self._load_molecule())

        # 分子信息
        self.mol_info = tk.Text(left, height=8, font=("Consolas", 9),
                                  bg="#f8f9fa", fg="#444", relief="flat", wrap="word")
        self.mol_info.pack(fill="x", padx=10, pady=10)

        # 加载按钮
        tk.Button(left, text="📥 加载分子", font=("Segoe UI", 10, "bold"),
                  bg="#2563eb", fg="white", relief="flat", pady=8,
                  command=self._load_molecule).pack(fill="x", padx=10, pady=5)

        # 中栏：3D可视化
        center = tk.Frame(frame, bg="#ffffff", relief="raised", bd=1)
        center.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        tk.Label(center, text="🧊 分子 3D 可视化", font=("Segoe UI", 12, "bold"),
                 bg="#ffffff", fg="#2563eb").pack(pady=10)

        self.fig = Figure(figsize=(6, 5), dpi=100, facecolor="#ffffff")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#fafafa")
        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

        # 控制按钮
        ctrl = tk.Frame(center, bg="#ffffff")
        ctrl.pack(fill="x", padx=10, pady=5)
        tk.Button(ctrl, text="🔄 重置视角", command=self._reset_view).pack(side="left", padx=5)
        tk.Button(ctrl, text="💾 导出图片", command=self._export_molecule_image).pack(side="left", padx=5)
        self.show_bonds = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="显示化学键", variable=self.show_bonds,
                       command=self._update_molecule_view, bg="#ffffff").pack(side="left", padx=10)

        # 右栏：自定义分子
        right = tk.Frame(frame, bg="#ffffff", width=300, relief="raised", bd=1)
        right.pack(side="left", fill="y", padx=5, pady=5)
        right.pack_propagate(False)

        tk.Label(right, text="✏️ 自定义分子", font=("Segoe UI", 12, "bold"),
                 bg="#ffffff", fg="#2563eb").pack(pady=10)

        tk.Label(right, text="原子坐标 (Symbol x y z, bohr)", font=("Segoe UI", 9),
                 bg="#ffffff", fg="#666").pack(anchor="w", padx=10)
        self.custom_atoms = scrolledtext.ScrolledText(right, height=12, font=("Consolas", 10),
                                                         bg="#f8f9fa", relief="flat")
        self.custom_atoms.pack(fill="x", padx=10, pady=5)
        self.custom_atoms.insert("1.0", "# 示例：水\nO 0 0 0\nH 1.107 1.430 0\nH -1.107 1.430 0")

        btn_row = tk.Frame(right, bg="#ffffff")
        btn_row.pack(fill="x", padx=10, pady=5)
        tk.Button(btn_row, text="解析", command=self._parse_custom).pack(side="left", padx=2)
        tk.Button(btn_row, text="清空", command=lambda: self.custom_atoms.delete("1.0", "end")).pack(side="left", padx=2)

        tk.Label(right, text="分子名称", font=("Segoe UI", 9), bg="#ffffff").pack(anchor="w", padx=10, pady=(10, 0))
        self.custom_name = tk.Entry(right, font=("Segoe UI", 10), relief="flat", bg="#f8f9fa")
        self.custom_name.pack(fill="x", padx=10, pady=5)
        self.custom_name.insert(0, "Custom Molecule")

        tk.Button(right, text="➕ 添加到库", font=("Segoe UI", 10, "bold"),
                  bg="#22c55e", fg="white", relief="flat", pady=8,
                  command=self._add_custom_molecule).pack(fill="x", padx=10, pady=10)

        # 初始化分子列表
        self._refresh_molecule_list()

    def _refresh_molecule_list(self):
        self.mol_listbox.delete(0, "end")
        filter_cat = self.mol_filter.get()
        for name, data in PRESET_MOLECULES.items():
            if filter_cat == "全部" or data["category"] == filter_cat:
                color = MOLECULE_CATEGORY_COLORS.get(data["category"], "#666")
                self.mol_listbox.insert("end", f"  {name}")
                idx = self.mol_listbox.size() - 1
                self.mol_listbox.itemconfig(idx, {"bg": "#fafafa"})

    def _on_molecule_select(self, event):
        selection = self.mol_listbox.curselection()
        if not selection:
            return
        name = self.mol_listbox.get(selection[0]).strip()
        if name in PRESET_MOLECULES:
            data = PRESET_MOLECULES[name]
            info = f"分子: {name}\n"
            info += f"电子数: {data['n_elec']}\n"
            info += f"分类: {data['category']}\n"
            info += f"原子数: {len(data['atoms'])}\n"
            info += f"赝势支持: {'/'.join([k for k,v in data['pp_supported'].items() if v])}\n"
            self.mol_info.delete("1.0", "end")
            self.mol_info.insert("1.0", info)

    def _load_molecule(self):
        selection = self.mol_listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "请先选择一个分子")
            return
        name = self.mol_listbox.get(selection[0]).strip()
        if name in PRESET_MOLECULES:
            self.current_molecule = name
            self.current_atoms = PRESET_MOLECULES[name]["atoms"]
            self._update_molecule_view()
            self._status(f"已加载: {name}")
            # 自动切换到配置页
            self.notebook.select(1)
            self._update_yaml_preview()

    def _parse_custom(self):
        text = self.custom_atoms.get("1.0", "end").strip()
        atoms = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    atoms.append({"symbol": parts[0], "coords": [float(parts[1]), float(parts[2]), float(parts[3])]})
                except ValueError:
                    continue
        if atoms:
            self.current_atoms = atoms
            self.current_molecule = self.custom_name.get() or "Custom"
            self._update_molecule_view()
            self._status(f"解析成功: {len(atoms)} 个原子")
        else:
            messagebox.showerror("错误", "无法解析原子坐标，请检查格式")

    def _add_custom_molecule(self):
        name = self.custom_name.get().strip()
        if not name:
            messagebox.showwarning("提示", "请输入分子名称")
            return
        text = self.custom_atoms.get("1.0", "end").strip()
        atoms = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    atoms.append({"symbol": parts[0], "coords": [float(parts[1]), float(parts[2]), float(parts[3])]})
                except ValueError:
                    continue
        if atoms:
            n_elec = sum({"H":1,"C":6,"N":7,"O":8,"F":9,"P":15,"S":16,"Cl":17}.get(a["symbol"], 0) for a in atoms)
            PRESET_MOLECULES[f"{name} ({n_elec}e)"] = {
                "atoms": atoms, "n_elec": n_elec, "category": "自定义",
                "pp_supported": {"AE": True, "ECP": True, "PH": False},
            }
            self._refresh_molecule_list()
            self._status(f"已添加: {name}")
        else:
            messagebox.showerror("错误", "无法解析原子坐标")

    def _update_molecule_view(self):
        self.ax.clear()
        if not self.current_atoms:
            self.ax.text(0.5, 0.5, 0.5, "请选择或加载分子", fontsize=14, ha="center", color="#999")
            self.canvas.draw()
            return

        coords = np.array([a["coords"] for a in self.current_atoms])
        symbols = [a["symbol"] for a in self.current_atoms]

        # 画原子
        for i, (sym, c) in enumerate(zip(symbols, coords)):
            color = ATOM_COLORS.get(sym, "#888888")
            size = ATOM_RADII.get(sym, 150)
            self.ax.scatter(c[0], c[1], c[2], c=color, s=size, edgecolors="#333", linewidths=0.5, zorder=5)
            self.ax.text(c[0], c[1], c[2]+0.3, sym, fontsize=9, ha="center", zorder=6)

        # 画化学键
        if self.show_bonds.get():
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist < BOND_THRESHOLD:
                        self.ax.plot([coords[i][0], coords[j][0]],
                                     [coords[i][1], coords[j][1]],
                                     [coords[i][2], coords[j][2]],
                                     color="#888", linewidth=2, zorder=3)

        self.ax.set_xlabel("X (bohr)", fontsize=8)
        self.ax.set_ylabel("Y (bohr)", fontsize=8)
        self.ax.set_zlabel("Z (bohr)", fontsize=8)
        self.ax.tick_params(labelsize=7)
        self.ax.set_title(f"{self.current_molecule} ({len(self.current_atoms)} atoms)", fontsize=10, pad=10)
        self.canvas.draw()

    def _reset_view(self):
        self.ax.view_init(elev=20, azim=45)
        self.canvas.draw()

    def _export_molecule_image(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            self._status(f"已导出: {path}")

    # ============================================================
    # 标签页2：配置生成（三栏布局 + 三档优化按钮 + YAML双向同步）
    # ============================================================
    def _build_tab_config(self):
        frame = tk.Frame(self.notebook, bg=THEMES["light"]["bg"])
        self.notebook.add(frame, text="  ⚙️ 配置生成  ")

        # 左栏：参数配置
        left = tk.Frame(frame, bg="#ffffff", width=380, relief="raised", bd=1)
        left.pack(side="left", fill="y", padx=5, pady=5)
        left.pack_propagate(False)

        tk.Label(left, text="⚙️ 计算配置", font=("Segoe UI", 12, "bold"),
                 bg="#ffffff", fg="#2563eb").pack(pady=10)

        # 当前分子显示
        mol_display = tk.Frame(left, bg="#f0f7ff", relief="flat", bd=1)
        mol_display.pack(fill="x", padx=10, pady=5)
        tk.Label(mol_display, text="当前分子:", font=("Segoe UI", 9),
                 bg="#f0f7ff", fg="#2563eb").pack(anchor="w", padx=10, pady=(5, 0))
        self.config_mol_label = tk.Label(mol_display, text="未选择", font=("Segoe UI", 11, "bold"),
                                           bg="#f0f7ff", fg="#202020")
        self.config_mol_label.pack(anchor="w", padx=10, pady=(0, 5))

        # 三档优化级别大按钮（借鉴东南·云霄）
        tk.Label(left, text="🎯 优化级别", font=("Segoe UI", 10, "bold"),
                 bg="#ffffff", fg="#333").pack(anchor="w", padx=10, pady=(10, 5))

        opt_frame = tk.Frame(left, bg="#ffffff")
        opt_frame.pack(fill="x", padx=10, pady=5)

        self.opt_buttons = {}
        for level_key, level_data in OPTIMIZATION_LEVELS.items():
            btn_frame = tk.Frame(opt_frame, bg="#ffffff", relief="raised", bd=2)
            btn_frame.pack(fill="x", pady=3)
            btn = tk.Radiobutton(btn_frame, variable=self.opt_level, value=level_key,
                                  text=level_data["name"], font=("Segoe UI", 11, "bold"),
                                  bg="#ffffff", fg=level_data["color"], selectcolor="#f0f0f0",
                                  anchor="w", command=self._on_opt_level_change)
            btn.pack(fill="x", padx=5, pady=(5, 0))
            tk.Label(btn_frame, text=level_data["desc"], font=("Segoe UI", 8),
                     bg="#ffffff", fg="#888", justify="left", anchor="w").pack(fill="x", padx=25, pady=(0, 5))
            self.opt_buttons[level_key] = btn_frame

        # 分隔线
        tk.Frame(left, height=1, bg="#e0e0e0").pack(fill="x", padx=10, pady=10)

        # 高级参数（可折叠）
        tk.Label(left, text="🔧 高级参数", font=("Segoe UI", 10, "bold"),
                 bg="#ffffff", fg="#333").pack(anchor="w", padx=10, pady=5)

        params_frame = tk.Frame(left, bg="#ffffff")
        params_frame.pack(fill="x", padx=10, pady=5)

        # 波函数架构
        tk.Label(params_frame, text="波函数架构", font=("Segoe UI", 9), bg="#ffffff").grid(row=0, column=0, sticky="w", pady=2)
        self.wf_arch = tk.StringVar(value="ferminet")
        ttk.Combobox(params_frame, textvariable=self.wf_arch, values=["ferminet", "psiformer"],
                      state="readonly", width=15).grid(row=0, column=1, sticky="e", pady=2)

        # 赝势
        tk.Label(params_frame, text="赝势类型", font=("Segoe UI", 9), bg="#ffffff").grid(row=1, column=0, sticky="w", pady=2)
        self.pp_type = tk.StringVar(value="AE")
        ttk.Combobox(params_frame, textvariable=self.pp_type, values=["AE", "ECP", "PH"],
                      state="readonly", width=15).grid(row=1, column=1, sticky="e", pady=2)

        # Forward Laplacian
        self.use_fl = tk.BooleanVar(value=True)
        tk.Checkbutton(params_frame, text="启用 Forward Laplacian", variable=self.use_fl,
                       bg="#ffffff", font=("Segoe UI", 9), command=self._update_yaml_preview).grid(row=2, column=0, columnspan=2, sticky="w", pady=2)

        # Sparse
        self.use_sparse = tk.BooleanVar(value=True)
        tk.Checkbutton(params_frame, text="启用 Sparse 导数", variable=self.use_sparse,
                       bg="#ffffff", font=("Segoe UI", 9), command=self._update_yaml_preview).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)

        # 手动参数
        tk.Label(params_frame, text="Batch Size", font=("Segoe UI", 9), bg="#ffffff").grid(row=4, column=0, sticky="w", pady=2)
        self.batch_size = tk.IntVar(value=2048)
        tk.Entry(params_frame, textvariable=self.batch_size, width=10, relief="flat", bg="#f8f9fa").grid(row=4, column=1, sticky="e", pady=2)

        tk.Label(params_frame, text="N_det", font=("Segoe UI", 9), bg="#ffffff").grid(row=5, column=0, sticky="w", pady=2)
        self.ndets = tk.IntVar(value=8)
        tk.Entry(params_frame, textvariable=self.ndets, width=10, relief="flat", bg="#f8f9fa").grid(row=5, column=1, sticky="e", pady=2)

        tk.Label(params_frame, text="训练步数", font=("Segoe UI", 9), bg="#ffffff").grid(row=6, column=0, sticky="w", pady=2)
        self.train_steps = tk.IntVar(value=5000)
        tk.Entry(params_frame, textvariable=self.train_steps, width=10, relief="flat", bg="#f8f9fa").grid(row=6, column=1, sticky="e", pady=2)

        # 保存路径
        tk.Label(params_frame, text="保存路径", font=("Segoe UI", 9), bg="#ffffff").grid(row=7, column=0, sticky="w", pady=2)
        self.save_path = tk.StringVar(value="./runs/calculation")
        tk.Entry(params_frame, textvariable=self.save_path, width=18, relief="flat", bg="#f8f9fa").grid(row=7, column=1, sticky="e", pady=2)

        # 操作按钮
        btn_frame = tk.Frame(left, bg="#ffffff")
        btn_frame.pack(fill="x", padx=10, pady=15)
        tk.Button(btn_frame, text="🔄 刷新YAML", font=("Segoe UI", 9),
                  bg="#f5f5f5", relief="flat", pady=6, command=self._update_yaml_preview).pack(fill="x", pady=2)
        tk.Button(btn_frame, text="💾 保存配置文件", font=("Segoe UI", 10, "bold"),
                  bg="#2563eb", fg="white", relief="flat", pady=8, command=self._save_config).pack(fill="x", pady=2)
        tk.Button(btn_frame, text="▶️ 开始计算", font=("Segoe UI", 10, "bold"),
                  bg="#22c55e", fg="white", relief="flat", pady=8, command=self._run_calculation).pack(fill="x", pady=2)

        # 中栏：3D可视化 + 分子信息
        center = tk.Frame(frame, bg="#ffffff", relief="raised", bd=1)
        center.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        tk.Label(center, text="🧊 分子预览", font=("Segoe UI", 12, "bold"),
                 bg="#ffffff", fg="#2563eb").pack(pady=10)

        self.config_fig = Figure(figsize=(5, 4), dpi=100, facecolor="#ffffff")
        self.config_ax = self.config_fig.add_subplot(111, projection="3d")
        self.config_ax.set_facecolor("#fafafa")
        self.config_canvas = FigureCanvasTkAgg(self.config_fig, master=center)
        self.config_canvas.draw()
        self.config_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

        # 分子信息卡片
        info_card = tk.Frame(center, bg="#f8f9fa", relief="flat")
        info_card.pack(fill="x", padx=10, pady=10)
        self.config_info_text = tk.Label(info_card, text="请先选择分子", font=("Segoe UI", 10),
                                           bg="#f8f9fa", fg="#666", justify="left", anchor="w")
        self.config_info_text.pack(fill="x", padx=10, pady=10)

        # 右栏：YAML实时预览（可编辑，双向同步）
        right = tk.Frame(frame, bg="#1e1e2e", width=420, relief="raised", bd=1)
        right.pack(side="left", fill="y", padx=5, pady=5)
        right.pack_propagate(False)

        # 标题栏
        title_bar = tk.Frame(right, bg="#313244", height=35)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)
        tk.Label(title_bar, text="📝 YAML 配置（可编辑）", font=("Consolas", 10, "bold"),
                 bg="#313244", fg="#cdd6f4").pack(side="left", padx=10)
        tk.Button(title_bar, text="↩️ 同步", font=("Segoe UI", 8),
                  bg="#45475a", fg="#cdd6f4", relief="flat", padx=8,
                  command=self._sync_yaml_to_form).pack(side="right", padx=5, pady=5)

        # YAML编辑器（深色主题，类似VS Code）
        self.yaml_editor = scrolledtext.ScrolledText(right, font=("Consolas", 10),
                                                        bg="#1e1e2e", fg="#cdd6f4",
                                                        insertbackground="#f5e0dc",
                                                        relief="flat", bd=0, wrap="none")
        self.yaml_editor.pack(fill="both", expand=True, padx=5, pady=5)
        self.yaml_editor.insert("1.0", "# 请先选择分子...")

        # 状态栏
        yaml_status = tk.Frame(right, bg="#313244", height=25)
        yaml_status.pack(fill="x")
        yaml_status.pack_propagate(False)
        self.yaml_status_var = tk.StringVar(value="就绪")
        tk.Label(yaml_status, textvariable=self.yaml_status_var, font=("Consolas", 8),
                 bg="#313244", fg="#a6adc8").pack(side="left", padx=10)

    def _on_opt_level_change(self):
        level = self.opt_level.get()
        config = OPTIMIZATION_LEVELS[level]
        self.batch_size.set(config["batch_size"])
        self.ndets.set(config["ndets"])
        self.train_steps.set(config["train_steps"])
        self._update_yaml_preview()
        self._status(f"已切换到: {config['name']} 模式")

    def _update_yaml_preview(self):
        if not self.current_atoms:
            self.yaml_editor.delete("1.0", "end")
            self.yaml_editor.insert("1.0", "# 请先在「分子构建」页选择分子...")
            return

        mol_name = self.current_molecule or "Unknown"
        self.config_mol_label.config(text=mol_name)

        # 更新3D预览
        self.config_ax.clear()
        coords = np.array([a["coords"] for a in self.current_atoms])
        symbols = [a["symbol"] for a in self.current_atoms]
        for sym, c in zip(symbols, coords):
            color = ATOM_COLORS.get(sym, "#888")
            size = ATOM_RADII.get(sym, 150)
            self.config_ax.scatter(c[0], c[1], c[2], c=color, s=size, edgecolors="#333", linewidths=0.5)
            self.config_ax.text(c[0], c[1], c[2]+0.3, sym, fontsize=8, ha="center")
        self.config_ax.set_xlabel("X", fontsize=7); self.config_ax.set_ylabel("Y", fontsize=7)
        self.config_ax.set_zlabel("Z", fontsize=7); self.config_ax.tick_params(labelsize=6)
        self.config_canvas.draw()

        # 更新信息卡片
        n_elec = sum({"H":1,"C":6,"N":7,"O":8,"F":9,"P":15,"S":16,"Cl":17}.get(a["symbol"], 0) for a in self.current_atoms)
        info = f"分子: {mol_name}\n原子数: {len(self.current_atoms)} | 电子数: {n_elec}\n"
        info += f"Batch: {self.batch_size.get()} | N_det: {self.ndets.get()} | 步数: {self.train_steps.get()}\n"
        info += f"FL: {'✓' if self.use_fl.get() else '✗'} | Sparse: {'✓' if self.use_sparse.get() else '✗'} | 赝势: {self.pp_type.get()}"
        self.config_info_text.config(text=info)

        # 生成YAML
        yaml = generate_yaml_config(
            mol_name=mol_name, atoms=self.current_atoms, pp_type=self.pp_type.get(),
            wf_arch=self.wf_arch.get(), ndets=self.ndets.get(),
            hidden_single=OPTIMIZATION_LEVELS[self.opt_level.get()]["hidden_single"],
            hidden_double=OPTIMIZATION_LEVELS[self.opt_level.get()]["hidden_double"],
            batch_size=self.batch_size.get(), pretrain_steps=OPTIMIZATION_LEVELS[self.opt_level.get()]["pretrain_steps"],
            train_steps=self.train_steps.get(), use_fl=self.use_fl.get(), use_sparse=self.use_sparse.get(),
            save_path=self.save_path.get(),
        )
        self.yaml_editor.delete("1.0", "end")
        self.yaml_editor.insert("1.0", yaml)
        self.yaml_status_var.set(f"已生成 ({len(yaml.splitlines())} 行)")

    def _sync_yaml_to_form(self):
        """从YAML编辑器同步到表单（双向同步）"""
        yaml_text = self.yaml_editor.get("1.0", "end")
        # 简单解析关键参数
        import re
        batch_match = re.search(r'batch_size:\s*(\d+)', yaml_text)
        ndets_match = re.search(r'ndets:\s*(\d+)', yaml_text)
        steps_match = re.search(r'iterations:\s*(\d+)', yaml_text)
        fl_match = re.search(r'use_forward_laplacian:\s*(true|false)', yaml_text)
        sparse_match = re.search(r'use_sparse_derivatives:\s*(true|false)', yaml_text)

        if batch_match: self.batch_size.set(int(batch_match.group(1)))
        if ndets_match: self.ndets.set(int(ndets_match.group(1)))
        if steps_match: self.train_steps.set(int(steps_match.group(1)))
        if fl_match: self.use_fl.set(fl_match.group(1) == "true")
        if sparse_match: self.use_sparse.set(sparse_match.group(1) == "true")

        self.yaml_status_var.set("已同步到表单")
        self._status("YAML已同步到表单参数")

    # ============================================================
    # 标签页3：计算管理
    # ============================================================
    def _build_tab_calculation(self):
        frame = tk.Frame(self.notebook, bg=THEMES["light"]["bg"])
        self.notebook.add(frame, text="  ▶️ 计算管理  ")

        # 顶部控制
        ctrl = tk.Frame(frame, bg="#ffffff", relief="raised", bd=1)
        ctrl.pack(fill="x", padx=5, pady=5)

        tk.Label(ctrl, text="当前任务:", font=("Segoe UI", 10, "bold"), bg="#ffffff").pack(side="left", padx=10, pady=10)
        self.current_task_label = tk.Label(ctrl, text="无", font=("Segoe UI", 10), bg="#ffffff", fg="#666")
        self.current_task_label.pack(side="left", padx=5)

        tk.Button(ctrl, text="▶️ 开始", font=("Segoe UI", 10, "bold"),
                  bg="#22c55e", fg="white", relief="flat", padx=15, pady=5,
                  command=self._run_calculation).pack(side="right", padx=5, pady=8)
        tk.Button(ctrl, text="⏹️ 停止", font=("Segoe UI", 10),
                  bg="#ef4444", fg="white", relief="flat", padx=15, pady=5,
                  command=self._stop_calculation).pack(side="right", padx=5, pady=8)

        # 进度条
        prog_frame = tk.Frame(frame, bg="#ffffff", relief="raised", bd=1)
        prog_frame.pack(fill="x", padx=5, pady=5)
        tk.Label(prog_frame, text="计算进度", font=("Segoe UI", 10), bg="#ffffff").pack(anchor="w", padx=10, pady=(5, 0))
        self.progress = ttk.Progressbar(prog_frame, mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=5)

        # 日志
        log_frame = tk.LabelFrame(frame, text="计算日志", bg="#ffffff", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, font=("Consolas", 9),
                                                     bg="#1e1e2e", fg="#a6e3a1", relief="flat")
        self.log_text.pack(fill="both", expand=True)

        # 底部按钮
        btn_frame = tk.Frame(frame, bg="#ffffff")
        btn_frame.pack(fill="x", padx=5, pady=5)
        tk.Button(btn_frame, text="📋 复制日志", command=self._copy_log).pack(side="left", padx=5)
        tk.Button(btn_frame, text="🗑️ 清空日志", command=lambda: self.log_text.delete("1.0", "end")).pack(side="left", padx=5)
        tk.Button(btn_frame, text="💾 保存日志", command=self._save_log).pack(side="left", padx=5)

    # ============================================================
    # 标签页4：批量队列
    # ============================================================
    def _build_tab_batch(self):
        frame = tk.Frame(self.notebook, bg=THEMES["light"]["bg"])
        self.notebook.add(frame, text="  📋 批量队列  ")

        # 左：任务列表
        left = tk.LabelFrame(frame, text="任务队列", bg="#ffffff", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        left.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # 工具栏
        toolbar = tk.Frame(left, bg="#ffffff")
        toolbar.pack(fill="x", pady=5)
        tk.Button(toolbar, text="➕ 添加当前配置", command=self._add_current_to_batch).pack(side="left", padx=2)
        tk.Button(toolbar, text="📂 批量添加配置文件", command=self._batch_add_configs).pack(side="left", padx=2)
        tk.Button(toolbar, text="🗑️ 删除选中", command=self._remove_selected_batch).pack(side="left", padx=2)
        tk.Button(toolbar, text="🧹 清除已完成", command=self.batch_manager.clear_completed).pack(side="left", padx=2)

        # 任务列表
        list_frame = tk.Frame(left, bg="#ffffff")
        list_frame.pack(fill="both", expand=True, pady=5)
        self.batch_listbox = tk.Listbox(list_frame, font=("Consolas", 10),
                                          bg="#fafafa", selectbackground="#2563eb",
                                          selectforeground="white", relief="flat")
        self.batch_listbox.pack(side="left", fill="both", expand=True)
        batch_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.batch_listbox.yview)
        batch_scroll.pack(side="right", fill="y")
        self.batch_listbox.config(yscrollcommand=batch_scroll.set)

        # 控制按钮
        ctrl_frame = tk.Frame(left, bg="#ffffff")
        ctrl_frame.pack(fill="x", pady=5)
        tk.Button(ctrl_frame, text="▶️ 开始批量计算", font=("Segoe UI", 11, "bold"),
                  bg="#22c55e", fg="white", relief="flat", pady=8,
                  command=self._start_batch).pack(fill="x", pady=2)
        tk.Button(ctrl_frame, text="⏹️ 停止", font=("Segoe UI", 10),
                  bg="#ef4444", fg="white", relief="flat", pady=6,
                  command=self.batch_manager.stop).pack(fill="x", pady=2)

        # 右：批量日志
        right = tk.LabelFrame(frame, text="批量计算日志", bg="#ffffff", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        right.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.batch_log = scrolledtext.ScrolledText(right, font=("Consolas", 9),
                                                      bg="#1e1e2e", fg="#a6e3a1", relief="flat")
        self.batch_log.pack(fill="both", expand=True)

    # ============================================================
    # 标签页5：结果分析
    # ============================================================
    def _build_tab_results(self):
        frame = tk.Frame(self.notebook, bg=THEMES["light"]["bg"])
        self.notebook.add(frame, text="  📊 结果分析  ")
        

        # 顶部：选择结果目录
        top = tk.Frame(frame, bg="#ffffff", relief="raised", bd=1)
        top.pack(fill="x", padx=5, pady=5)
        tk.Label(top, text="结果目录:", font=("Segoe UI", 10), bg="#ffffff").pack(side="left", padx=10, pady=10)
        self.result_dir = tk.StringVar(value="./runs")
        tk.Entry(top, textvariable=self.result_dir, width=40, relief="flat", bg="#f8f9fa").pack(side="left", padx=5)
        tk.Button(top, text="📂 浏览", command=self._browse_result_dir).pack(side="left", padx=5)
        tk.Button(top, text="🔄 加载", font=("Segoe UI", 10, "bold"),
                  bg="#2563eb", fg="white", relief="flat", padx=15, pady=5,
                  command=self._load_results).pack(side="left", padx=10)

        # 左：能量统计
        left = tk.LabelFrame(frame, text="能量统计", bg="#ffffff", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        left.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.stats_text = scrolledtext.ScrolledText(left, font=("Consolas", 10),
                                                       bg="#f8f9fa", relief="flat", height=10)
        self.stats_text.pack(fill="both", expand=True)

        # 右：收敛曲线
        right = tk.LabelFrame(frame, text="能量收敛曲线", bg="#ffffff", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        right.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.result_fig = Figure(figsize=(6, 4), dpi=100, facecolor="#ffffff")
        self.result_ax = self.result_fig.add_subplot(111)
        self.result_ax.set_facecolor("#fafafa")
        self.result_canvas = FigureCanvasTkAgg(self.result_fig, master=right)
        self.result_canvas.draw()
        self.result_canvas.get_tk_widget().pack(fill="both", expand=True)

        # 底部：导出按钮
        bottom = tk.Frame(frame, bg="#ffffff")
        bottom.pack(fill="x", padx=5, pady=5)
        tk.Button(bottom, text="📈 导出收敛曲线 (PNG)", command=self._export_convergence_plot).pack(side="left", padx=5)
        tk.Button(bottom, text="📊 导出数据表格 (CSV)", command=self._export_stats_csv).pack(side="left", padx=5)
        tk.Button(bottom, text="📄 生成论文图表", command=self._generate_paper_figures).pack(side="left", padx=5)

    # ============================================================
    # 回调函数
    # ============================================================
    def _log(self, text):
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")

    def _batch_log(self, text):
        self.batch_log.insert("end", text + "\n")
        self.batch_log.see("end")

    def _status(self, text):
        self.status_var.set(text)

    def _calc_finished(self, rc):
        self.progress.stop()
        self._status(f"计算完成 (exit {rc})")
        messagebox.showinfo("完成", f"计算已完成！\n退出码: {rc}")

    def _update_batch_list(self):
        self.batch_listbox.delete(0, "end")
        status_colors = {"等待": "#888", "运行中": "#2563eb", "完成": "#22c55e", "错误": "#ef4444"}
        for i, task in enumerate(self.batch_manager.tasks):
            status = task["status"]
            color = status_colors.get(status, "#333")
            self.batch_listbox.insert("end", f"  [{i+1:02d}] {task['name']:<40s} {status}")
            self.batch_listbox.itemconfig(i, fg=color)

    def _run_calculation(self):
        if not self.current_atoms:
            messagebox.showwarning("提示", "请先选择分子")
            return
        # 保存临时配置
        yaml_path = os.path.join(self.work_dir.get(), "temp_config.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(self.yaml_editor.get("1.0", "end"))
        self.current_task_label.config(text=self.current_molecule)
        self.progress.start(10)
        self.calc_manager.start(yaml_path, self.jaqmc_path.get(), self.conda_lib_path.get())
        self.notebook.select(2)

    def _stop_calculation(self):
        self.calc_manager.stop()
        self.progress.stop()

    def _add_current_to_batch(self):
        if not self.current_atoms:
            messagebox.showwarning("提示", "请先选择分子")
            return
        yaml_path = os.path.join(self.work_dir.get(), f"batch_{self.current_molecule.replace(' ', '_')}.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(self.yaml_editor.get("1.0", "end"))
        self.batch_manager.add_task(self.current_molecule, yaml_path)
        self._status(f"已添加到批量队列: {self.current_molecule}")

    def _batch_add_configs(self):
        files = filedialog.askopenfilenames(filetypes=[("YAML", "*.yaml *.yml")])
        for f in files:
            name = os.path.basename(f)
            self.batch_manager.add_task(name, f)
        self._status(f"已添加 {len(files)} 个任务")

    def _remove_selected_batch(self):
        selection = self.batch_listbox.curselection()
        if selection:
            self.batch_manager.remove_task(selection[0])

    def _start_batch(self):
        if not self.batch_manager.tasks:
            messagebox.showwarning("提示", "队列为空")
            return
        self.batch_manager.start(self.jaqmc_path.get(), self.conda_lib_path.get())
        self.notebook.select(3)

    def _load_benchmark_panel(self):
        """一键加载16分子基准面板到批量队列"""
        count = 0
        for name, data in PRESET_MOLECULES.items():
            self.current_molecule = name
            self.current_atoms = data["atoms"]
            self.pp_type.set(data["pp_supported"].get("PH") and "PH" or "AE")
            self._update_yaml_preview()
            yaml_path = os.path.join(self.work_dir.get(), f"benchmark_{name.replace(' ', '_').replace('(', '').replace(')', '')}.yaml")
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(self.yaml_editor.get("1.0", "end"))
            self.batch_manager.add_task(name, yaml_path)
            count += 1
        self._status(f"已加载16分子基准面板 ({count}个任务)")
        messagebox.showinfo("完成", f"已加载16分子基准面板到批量队列！\n共 {count} 个任务")

    def _browse_result_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.result_dir.set(path)

    def _load_results(self):
        # 简化版：查找train_stats.csv
        stats_dir = self.result_dir.get()
        if not os.path.exists(stats_dir):
            messagebox.showerror("错误", "目录不存在")
            return
        csv_files = list(Path(stats_dir).rglob("*train_stats*.csv"))
        if not csv_files:
            messagebox.showwarning("提示", "未找到结果文件")
            return
        # 读取第一个
        import pandas as pd
        df = pd.read_csv(csv_files[0])
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", df.describe().to_string())
        # 画图
        self.result_ax.clear()
        if "energy" in df.columns:
            self.result_ax.plot(df["energy"], label="Energy", color="#2563eb")
            self.result_ax.set_xlabel("Step")
            self.result_ax.set_ylabel("Energy (Ha)")
            self.result_ax.legend()
            self.result_ax.grid(True, alpha=0.3)
        self.result_canvas.draw()
        self._status(f"已加载: {csv_files[0].name}")

    def _export_convergence_plot(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.result_fig.savefig(path, dpi=300, bbox_inches="tight")
            self._status(f"已导出: {path}")

    def _export_stats_csv(self):
        messagebox.showinfo("提示", "功能开发中...")

    def _generate_paper_figures(self):
        messagebox.showinfo("提示", "论文图表生成功能开发中...\n将支持：收敛曲线、能量对比、加速比分析")

    def _copy_log(self):
        self.clipboard_clear()
        self.clipboard_append(self.log_text.get("1.0", "end"))
        self._status("日志已复制到剪贴板")

    def _save_log(self):
        path = filedialog.asksaveasfilename(defaultextension=".log", filetypes=[("Log", "*.log"), ("Text", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.log_text.get("1.0", "end"))
            self._status(f"日志已保存: {path}")

    def _open_config(self):
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml")])
        if path:
            with open(path, "r", encoding="utf-8") as f:
                self.yaml_editor.delete("1.0", "end")
                self.yaml_editor.insert("1.0", f.read())
            self._status(f"已打开: {path}")
            self.notebook.select(1)

    def _save_config(self):
        path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML", "*.yaml")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.yaml_editor.get("1.0", "end"))
            self._status(f"已保存: {path}")

    def _export_results(self):
        messagebox.showinfo("提示", "请在「结果分析」页导出")

    def _goto_results(self):
        self.notebook.select(4)

    def _env_settings(self):
        win = tk.Toplevel(self)
        win.title("环境设置")
        win.geometry("500x250")
        win.configure(bg="#f5f5f5")
        win.transient(self)
        win.grab_set()

        tk.Label(win, text="JaQMC 路径:", font=("Segoe UI", 10), bg="#f5f5f5").pack(anchor="w", padx=20, pady=(20, 5))
        tk.Entry(win, textvariable=self.jaqmc_path, width=50, relief="flat", bg="#ffffff").pack(padx=20)

        tk.Label(win, text="Conda Library 路径:", font=("Segoe UI", 10), bg="#f5f5f5").pack(anchor="w", padx=20, pady=(15, 5))
        tk.Entry(win, textvariable=self.conda_lib_path, width=50, relief="flat", bg="#ffffff").pack(padx=20)

        tk.Label(win, text="工作目录:", font=("Segoe UI", 10), bg="#f5f5f5").pack(anchor="w", padx=20, pady=(15, 5))
        tk.Entry(win, textvariable=self.work_dir, width=50, relief="flat", bg="#ffffff").pack(padx=20)

        tk.Button(win, text="确定", font=("Segoe UI", 10, "bold"),
                  bg="#2563eb", fg="white", relief="flat", padx=20, pady=6,
                  command=win.destroy).pack(pady=20)

    def _toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        t = THEMES[self.theme]
        self.configure(bg=t["bg"])
        self._status(f"已切换到{'深色' if self.theme == 'dark' else '浅色'}主题")

    def _show_onboarding(self):
        OnboardingWindow(self)

    def _about(self):
        messagebox.showinfo("关于",
            "NN-VMC Workbench v2.1\n\n"
            "神经网络变分蒙特卡洛桌面工作台\n"
            "基于 JaQMC (JAX) 框架\n\n"
            "功能：分子构建 | 配置生成 | 计算管理 | 批量队列 | 结果分析\n\n"
            "v2.1 改进：三栏布局 | 三档优化级别 | 分子库彩色分类 | YAML双向同步 | 新手引导\n\n"
            "© 2026 NN-VMC Workbench Team")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    app = NNVMWorkbench()
    app.mainloop()
