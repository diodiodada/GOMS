实验计划：

state: 10: 3 position, 3 linear velocity, 4 angular velocity
action: 3 position control
goal: 3 position


实验1：    输入 state + goal
          输出 action
          one-step version


问题1：
        怎么让 random 采集到的数据 make sense ?!!
答案1：
        手动编写控制代码

问题2：
        训练得到的 policy 的成功率为0
答案2：
        让原来的回归任务改成分类任务
        训练4个分类器，其中3个是3分类任务，一个是2分类任务

        上面的答案不对！！
        正确的原因是提供的特征的维度太多，造成了维度爆炸。
        解决办法是把原来的长度为25的observation缩短到长度为6

        结果：
        无论是分类任务还是回归任务，准确率都可以接近100%