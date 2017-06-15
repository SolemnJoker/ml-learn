#-*- coding: UTF-8 -*-
import numpy as np
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

myfont = fm.FontProperties(
    fname=matplotlib.font_manager.win32FontDirectory() + "/STXIHEI.TTF", size=14)
matplotlib.rcParams["axes.unicode_minus"] = False


#%%
def simple_plot():
    """
    simple plot
    """
    # 生成测试数据
    x = np.linspace(-np.pi, np.pi, 256)
    y_sin, y_cos = np.sin(x), np.cos(x)

    # 生成画布
    plt.figure(figsize=(8, 6), dpi=80)
    plt.title("可视化标题", fontproperties=myfont)
    plt.grid(True)

    # 设置X轴
    plt.xlabel("X轴", fontproperties=myfont)
    plt.xlim(-4.0, 4.0)
    plt.xticks(np.linspace(-4, 4, 9))

    # 设置Y轴
    plt.ylabel("Y轴", fontproperties=myfont)
    plt.ylim(-1.0, 1.0)
    plt.yticks(np.linspace(-1, 1, 9))

    plt.plot(x, y_cos, "b--", linewidth=2.0, label="cos")
    plt.plot(x, y_sin, "g-", linewidth=2.0, label="sin")
    # 设置图例位置,loc可以为[upper, lower, left, right, center]
    plt.legend(loc="upper left", prop=myfont, shadow=True)
    plt.show()
    return


simple_plot()

#%%


def simple_advanced_plot():
    """
    simple advanced plot
    """
    # 生成测试数据
    x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    y_cos, y_sin = np.cos(x), np.sin(x)

    # 生成画布, 并设定标题
    plt.figure(figsize=(8, 6), dpi=80)
    plt.title("simple_advanced_plot", fontproperties=myfont)
    plt.grid(True)

    # 画图的另外一种方式
    ax_1 = plt.subplot(111)
    ax_1.plot(x, y_cos, color="#0000ff", linewidth=2.0,
              linestyle="--", label="左cos")
    ax_1.legend(loc="upper left", prop=myfont, shadow=True)

    # 设置Y轴(左边)
    ax_1.set_ylabel("左cos的y轴", fontproperties=myfont)
    ax_1.set_ylim(-1.0, 1.0)
    ax_1.set_yticks(np.linspace(-1, 1, 9, endpoint=True))

    # 画图的另外一种方式
    ax_2 = ax_1.twinx()
    ax_2.plot(x, y_sin, color="green", linewidth=2.0,
              linestyle="-", label="右sin")
    ax_2.legend(loc="upper right", prop=myfont, shadow=True)

    # 设置Y轴(右边)
    ax_2.set_ylabel("右sin的y轴", fontproperties=myfont)
    ax_2.set_ylim(-2.0, 2.0)
    ax_2.set_yticks(np.linspace(-2, 2, 9, endpoint=True))

    # 设置X轴(共同)
    ax_1.set_xlabel("x轴", fontproperties=myfont)
    ax_1.set_xlim(-4.0, 4.0)
    ax_1.set_xticks(np.linspace(-4, 4, 9, endpoint=True))

    # 图形显示
    plt.show()
    return


simple_advanced_plot()
