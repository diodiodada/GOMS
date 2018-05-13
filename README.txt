实验计划：

state: 10: 3 position, 3 linear velocity, 4 angular velocity
action: 3 position control
goal: 3 position


实验1：    训练：
          输入 state + next_state
          输出 action
          测试：
          输入 state + goal
          输出 action
结果1：    对于 fetch_reacher 任务来说效果不错

实验2：    训练：
          输入 state + goal
          输出 action
          测试：
          输入 state + goal
          输出 action
结果2：    对于 fetch_pick_and_place 任务来说效果不错

实验3：    在实验2上减少训练的样本数量


问题1：怎么让 random 采集到的数据 make sense ?!!
答案1：手动编写控制代码

问题2：训练得到的 policy 的成功率为0
答案2：让原来的回归任务改成分类任务
      训练4个分类器，其中3个是3分类任务，一个是2分类任务

      上面的答案不对！！
      正确的原因是提供的特征的维度太多，造成了维度爆炸。
      解决办法是把原来的长度为25的observation缩短到长度为6

      结果：
      无论是分类任务还是回归任务，准确率都可以接近100%

问题3：如何从gym中得到图片
答案3：
        1. 设置render的mode参数
            env.render(mode='rgb_array')
        2. 去除多余的显示信息
            ~/Documents/mujoco-py/mujoco_py/mjviewer.py中
           class MjViewer 中
           self._hide_overlay = True
        3. 更改图片的大小
            ~/anaconda3/envs/tf-cpu/lib/python3.6/site-packages/gym/envs/mujoco/mujoco_env.py中
           class:MujocoEnv function:render中
           width, height = 1744, 992
