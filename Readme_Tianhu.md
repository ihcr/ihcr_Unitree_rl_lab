# Unitree RL Lab — G1 Sim2Real 部署指南

本项目支持在 **Mujoco 仿真** 与 **Unitree G1 真机** 上部署并运行策略（policy）。  
本文提供从编译到运行的完整步骤、按键说明、策略切换方法与常见网络问题排查。

---

## 目录
- [准备与编译](#准备与编译)
- [快速开始](#快速开始)
  - [在 Mujoco 中部署](#在-mujoco-中部署)
  - [在 Unitree G1 真机上部署](#在-unitree-g1-真机上部署)
- [运行界面按键](#运行界面按键)
- [切换/更新策略](#切换更新策略)
- [（可选）导出部署配置 `deploy.yaml`](#可选导出部署配置-deployyaml)
- [网络接口与问题排查](#网络接口与问题排查)
- [安全注意事项](#安全注意事项)

---

## 准备与编译

确保训练环境（Isaac Lab/策略训练脚本）可正常运行，并已拉取本仓库代码。

编译真机/仿真控制端：
```bash
cd unitree_rl_lab/deploy/robots/g1
mkdir -p build && cd build
cmake ..
make -j
```

> 若已编译成功，可直接跳过本节。

---

## 快速开始

### 在 Mujoco 中部署

**终端 A — 启动仿真器**
```bash
cd unitree_mujoco/simulate/build
./unitree_mujoco
```

**终端 B — 启动控制器（使用回环网卡）**
```bash
cd unitree_rl_lab/deploy/robots/g1/build
./g1_ctrl --network lo
```

> 说明：仿真场景下使用 `lo`（loopback）即可。

---

### 在 Unitree G1 真机上部署

> **上机前务必确认：**
> 1. 机器人板载控制程序**已关闭**（避免控制冲突）。  
> 2. 上位机与机器人通过**有线直连**，并在**同一网段**（必要时为上位机配置静态 IP）。  
> 3. 选择正确的**有线网卡名**（如 `eno2`、`enp3s0` 等，见下文“网络接口与问题排查”）。

启动控制器：
```bash
cd unitree_rl_lab/deploy/robots/g1/build
./g1_ctrl --network eno2
# 如需管理员权限：
# sudo ./g1_ctrl --network eno2
```

---

## 运行界面按键

在 `g1_ctrl` 的终端界面中：

- **`s`**：进入/保持 **Fixed Stand**（固定站立）模式  
- **`v`**：**部署并运行策略（policy）**  
- **`q`**：退出界面

---

## 切换/更新策略

编辑部署侧 `config.yaml`，将 `policy_dir:` 指向你的新策略目录，然后重新运行 `g1_ctrl`：

```yaml
# config.yaml
policy_dir: /path/to/your/policy_dir
```

> 如果新策略需要**新增观测**，请在部署侧的 `observation.h` 中实现对应观测项并**重新编译**。

---

## （可选）导出部署配置 `deploy.yaml`

训练端可导出包含关节映射、控制/观测参数等的部署配置，以在真机侧保持一致。示例（根据你的项目结构调整导入路径）：

```python
# 在你的训练/评估脚本中
from your_module import export_deploy_cfg  # 替换为实际路径

# env: ManagerBasedRLEnv 实例
# log_dir: 你的日志目录
export_deploy_cfg(env, log_dir)

# 将在 {log_dir}/params/deploy.yaml 生成部署配置（关节映射、PD参数、观测配置等）
```

将生成的 `deploy.yaml` 放到部署端程序期望读取的位置（若有固定查找路径请保持一致）。

---

## 网络接口与问题排查

**查看本机可用网卡名：**
```bash
ip -o link show | awk -F': ' '{print $2}' | cut -d'@' -f1
```
典型输出：
```
lo
eno2
wlp0s20f3
docker0
...
```
选择其中的**有线网卡**作为参数传给 `--network`（示例：`--network eno2`）。

**常见错误与修复：**

- 报错：`<iface>: does not match an available interface.`  
  **原因**：传入的网卡名不存在或拼写错误。  
  **处理**：用上面命令确认网卡名，替换为正确的接口。

- 报错：与 DDS/CycloneDDS 相关（如 `Failed to create domain`）  
  **常见原因**：网卡/网络配置异常、权限不足、防火墙限制或同网段不匹配。  
  **处理建议**：  
  1) 确保选择的是**真实可用的有线网卡**；  
  2) 尝试使用 `sudo` 运行；  
  3) 关闭防火墙或放通相关端口；  
  4) 确保与机器人在**同一网段**；  
  5) 确认机器人板载控制程序**已关闭**，避免冲突。

---

## 安全注意事项

- 初次上机请在空旷、安全区域进行，并准备好**急停**与**保护措施**。  
- 建议先进入 **Fixed Stand** 模式观察关节稳定性，再切换到策略运行。  
- 更新策略/观测/控制参数后，务必**重新编译**并以**低速、小幅度**逐步验证。

---

如需扩展观测或动作空间，请在部署端相应源码（如 `observation.h`、动作配置）中实现并重新编译