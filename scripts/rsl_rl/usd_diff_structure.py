# usd_diff_structure.py
from pxr import Usd

def list_prim_paths(usd_path):
    stage = Usd.Stage.Open(usd_path)
    return set([str(prim.GetPath()) for prim in stage.Traverse()])

def main():
    # 修改为你的两个USD路径
    usd_file_1 = "/home/tianhup/Desktop/unitree_rl_lab/unitree_model/G1/g1.usd"
    usd_file_2 = "/home/tianhup/Desktop/unitree_rl_lab/unitree_model/G1/g1_minimal.usd"

    prims_1 = list_prim_paths(usd_file_1)
    prims_2 = list_prim_paths(usd_file_2)

    only_in_1 = prims_1 - prims_2
    only_in_2 = prims_2 - prims_1

    print(f"\n🔍 Prims only in {usd_file_1}:")
    for p in sorted(only_in_1):
        print(" -", p)

    print(f"\n🔍 Prims only in {usd_file_2}:")
    for p in sorted(only_in_2):
        print(" -", p)

if __name__ == "__main__":
    main()

