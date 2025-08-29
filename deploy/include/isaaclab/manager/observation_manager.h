// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <unordered_set>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <cstdio>      // std::snprintf

#include <spdlog/spdlog.h>

#include "isaaclab/manager/manager_term_cfg.h"

namespace isaaclab
{

// ====== 观测项注册表 ======
using ObsMap = std::map<std::string, ObsFunc>;

inline ObsMap& observations_map() {
    static ObsMap instance;
    return instance;
}

#define REGISTER_OBSERVATION(name) \
    inline std::vector<float> name(ManagerBasedRLEnv* env); \
    inline struct name##_registrar { \
        name##_registrar() { observations_map()[#name] = name; } \
    } name##_registrar_instance; \
    inline std::vector<float> name(ManagerBasedRLEnv* env)

// ====== ObservationManager ======
class ObservationManager
{
public:
    ObservationManager(YAML::Node cfg, ManagerBasedRLEnv* env)
    : cfg(cfg), env(env)
    {
        _prepare_terms();
    }

    void reset()
    {
        for (auto& term : obs_term_cfgs)
        {
            term.reset(term.func(this->env));
        }
    }

    std::vector<float> compute()
    {
        std::vector<float> obs;
        obs.reserve(640); // 预留一点空间

        // 为本次 compute 动态记录每个 term 的起始偏移和长度
        struct TermSpan { std::string name; int start; int length; };
        std::vector<TermSpan> spans; spans.reserve(obs_term_cfgs.size());

        for (size_t i = 0; i < obs_term_cfgs.size(); ++i)
        {
            auto& term = obs_term_cfgs[i];
            const int start = static_cast<int>(obs.size());

            // 更新历史
            term.add(term.func(this->env));
            // 取已 scale/clip 的拼接结果
            auto term_obs_scaled = term.get();

            const int len = static_cast<int>(term_obs_scaled.size());
            obs.insert(obs.end(), term_obs_scaled.begin(), term_obs_scaled.end());

            spans.push_back(TermSpan{term_names_[i], start, len});
        }

        // ==== DEBUG: 仅打印前若干次，且按名字精确切片 ====
        static int dbg_obs_count = 0;
        constexpr int DBG_MAX_PRINTS = 10;

        if (dbg_obs_count < DBG_MAX_PRINTS)
        {
            auto dump_slice = [&](int off, int len){
                std::string s; s.reserve(len * 14);
                for (int i = 0; i < len; ++i) {
                    char buf[64];
                    std::snprintf(buf, sizeof(buf), "%.6f", obs[off + i]);
                    s += buf;
                    if (i + 1 < len) s += ", ";
                }
                return s;
            };

            // 小助手：按名字安全打印某个 term（若不存在则跳过并给出提示）
            auto print_span = [&](const std::string& name)
            {
                for (const auto& sp : spans) {
                    if (sp.name == name) {
                        spdlog::info("[C++] {} ({}): [{}]", name, sp.length, dump_slice(sp.start, sp.length));
                        return;
                    }
                }
                spdlog::warn("[C++] term '{}' not found in this observation layout", name);
            };

            spdlog::info("[C++] OBS#{} TOTAL={} dims", dbg_obs_count, (int)obs.size());

            // 按名字精确打印（跟 YAML 顺序无关）
            // print_span("base_ang_vel");
            // print_span("projected_gravity");
            // print_span("velocity_commands");
            // print_span("joint_pos_rel");
            // print_span("joint_vel_rel");
            // print_span("last_action");
            // print_span("curr_demo_dof_pos");
            // print_span("next_demo_dof_pos");

            ++dbg_obs_count;
        }
        // ==== DEBUG 结束 ====

        return obs;
    }

protected:
    void _prepare_terms()
    {
        term_names_.clear();
        obs_term_cfgs.clear();
        term_names_.reserve(this->cfg.size());
        obs_term_cfgs.reserve(this->cfg.size());

        // 逐项解析 YAML，构建 ObservationTermCfg
        for (auto it = this->cfg.begin(); it != this->cfg.end(); ++it)
        {
            const std::string term_name = it->first.as<std::string>();
            auto term_yaml_cfg = it->second;

            ObservationTermCfg term_cfg;

            // 历史长度
            term_cfg.history_length = term_yaml_cfg["history_length"].as<int>(1);

            // 函数指针
            auto itObs = observations_map().find(term_name);
            if (itObs == observations_map().end() || itObs->second == nullptr)
            {
                throw std::runtime_error("Observation term '" + term_name + "' is not registered.");
            }
            term_cfg.func = itObs->second;

            // 预热：用当前值初始化环形缓冲
            auto init_obs = term_cfg.func(this->env);
            term_cfg.reset(init_obs);

            // scale 必须配置；若要默认值，这里可给 fallback
            if (term_yaml_cfg["scale"].IsDefined() && term_yaml_cfg["scale"].IsSequence()) {
                term_cfg.scale = term_yaml_cfg["scale"].as<std::vector<float>>();
            } else {
                throw std::runtime_error("Observation term '" + term_name + "' requires 'scale' in YAML.");
            }

            // 可选 clip
            if (!term_yaml_cfg["clip"].IsNull()) {
                term_cfg.clip = term_yaml_cfg["clip"].as<std::vector<float>>();
            }

            // 入表：配置与名字并行保存
            obs_term_cfgs.push_back(term_cfg);
            term_names_.push_back(term_name);
        }
    }

    YAML::Node cfg;
    ManagerBasedRLEnv* env;

private:
    std::vector<ObservationTermCfg> obs_term_cfgs;
    std::vector<std::string>        term_names_;
};

} // namespace isaaclab
