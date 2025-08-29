#include "unitree/dds_wrapper/robots/g1/g1.h"   // ✅ 用 g1
#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"

#include <thread>
#include <atomic>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>   // std::exit
#include <spdlog/spdlog.h>

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(200000);
    if (!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        std::exit(EXIT_FAILURE);  // ✅ 直接退出（SDK无g1::shutdown / ChannelFactory::Shutdown）
    }

    FSMState::lowcmd   = std::make_unique<unitree::robot::g1::publisher::LowCmd>();
    FSMState::lowstate = std::make_shared<unitree::robot::g1::subscription::LowState>();

    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

enum FSMMode { Passive = 1, FixStand = 2, Velocity = 3 };

int main(int argc, char** argv)
{
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();
    FSMState::lowcmd->msg_.mode_machine() = 5; // 29DoF

    // ---- 键盘：s=>FixStand, v=>Velocity, q=>退出 ----
    std::atomic_bool key_fixstand{false}, key_velocity{false}, key_quit{false};
    auto keyboard_thread = std::thread([&](){
        termios oldt{};
        tcgetattr(STDIN_FILENO, &oldt);
        termios newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        int oldfl = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, oldfl | O_NONBLOCK);

        printf("[KEYS] s: FixStand, v: Velocity, q: quit\n");
        while (!key_quit.load()) {
            char c;
            if (read(STDIN_FILENO, &c, 1) == 1) {
                if (c == 's') { key_fixstand = true;  printf("[KEYS] FixStand requested\n"); }
                else if (c == 'v') { key_velocity = true; printf("[KEYS] Velocity requested\n"); }
                else if (c == 'q') { key_quit = true;     printf("[KEYS] Quit requested\n"); }
            }
            usleep(10000);
        }
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        fcntl(STDIN_FILENO, F_SETFL, oldfl);
    });

    auto& joy = FSMState::lowstate->joystick;

    // 初始：Passive
    auto fsm = std::make_unique<CtrlFSM>(new State_Passive(FSMMode::Passive));
    auto* st_passive = fsm->states.back();  // 方便在 Passive 上注册多个触发

    // Passive --(L2+Up 或 's')-> FixStand
    st_passive->registered_checks.emplace_back(std::make_pair(
        [&]()->bool{
            bool trig = key_fixstand.exchange(false);
            return (joy.LT.pressed && joy.up.on_pressed) || trig;
        },
        (int)FSMMode::FixStand));

    // ✅ 新增：Passive --(RB+X 或 'v')-> Velocity（跳过 FixStand）
    st_passive->registered_checks.emplace_back(std::make_pair(
        [&]()->bool{
            bool trig = key_velocity.exchange(false);
            return (joy.RB.pressed && joy.X.on_pressed) || trig;
        },
        (int)FSMMode::Velocity));

    // FixStand 状态（保留老路径）
    fsm->add(new State_FixStand(FSMMode::FixStand));

    // （可保留）FixStand --(RB+X 或 'v')-> Velocity
    fsm->states.back()->registered_checks.emplace_back(std::make_pair(
        [&]()->bool{
            bool trig = key_velocity.exchange(false);
            return (joy.RB.pressed && joy.X.on_pressed) || trig;
        },
        (int)FSMMode::Velocity));

    // Velocity（策略状态）
    fsm->add(new State_RLBase(FSMMode::Velocity, "Velocity"));

    // 提示
    std::cout << "Press 'v' to run the policy (skip FixStand).\n";
    std::cout << "Or press 's' for FixStand, then 'v'.\n";
    std::cout << "Press 'q' in this terminal to quit.\n";

    // ✅ 可退出主循环
    while (!key_quit.load()) {
        sleep(1);
    }
    if (keyboard_thread.joinable()) keyboard_thread.join();

    return 0;
}