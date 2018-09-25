import numpy as np
# 导入之前定义的Cluster类以计算各种信息量
from CvDTree.Cluster import Cluster


# 定义一个足够抽象的基类已囊括所有我们关心的算法
class CvDNode:
    """
        初始化结构
        self._x,self._y:记录数据集的变量
        self.base,self.chaos:记录对数的底和当前的不确定性
        self.criterion,self.category:记录该Node计算信息增益的方法和所属的类别
        self.left_child,self.right_child:针对连续型特征和CART、记录下该Node的左右子节点
        self._children,self.leafs:记录该Node的所有子节点和所有下属的叶节点
        self.sample_weight:记录样本权重
        self.wc:记录各个维度的特征是否连续的列表（whether continuous）
        self.tree:记录该Node所属的Tree
        self.feature_dim,self.tar,self.feats:记录该Node划分标准的相关信息。具体而言：
            self.feature_dim:记录作为划分标准的特征所对应的维度j*
            self.tar:针对连续型特征和CART，记录二分标准
            self.feats:记录该Node能进行选择的、作为划分标准的特征的维度
        self.parent,self.is_root:记录该Node的父节点以及该Node是否为根节点
        self._depth,self.prev_feat:记录Node的深度和其父节点的划分标准
        self.is_cart:记录该Node是否使用了CART算法
        self.is_continuous:记录该Node选择的划分标准对应的特征是否连续
        self.pruned:记录该Node是否已被剪掉，后面实现局部剪枝算法会用到
    """

    def __init__(self,
                 tree=None,
                 base=2,
                 chaos=None,
                 depth=0,
                 parent=None,
                 is_root=True,
                 prev_feat="Root"):
        self._x = self._y = None
        self.base, self.chaos = base, chaos
        self.criterion = self.category = None
        self.left_child = self.right_child = None
        self._children, self.leafs = {}, {}
        self.sample_weight = None
        self.wc = None
        self.tree = tree
        # 如果传入了Tree的话就进行相应的初始化
        if tree is not None:
            # 由于数据预处理是由Tree完成的
            # 所以各个维度的特征是否是连续型随机变量也是由Tree记录的
            self.wc = tree.whether_continuous
            # 这里的nodes变量是Tree中记录所有Node的列表
            tree.nodes.appdend(self)
        self.feature_dim, self.tar, self.feats = None, None, []
        self.parent, self.is_root = parent, is_root
        self._depth, self.prev_feat = depth, prev_feat
        self.is_cart = self.is_continuous = self.pruned = False

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    # 重载__lt__方法，使得Node之间可以比较谁更小、进而方便调试和可视化
    def __lt__(self, other):
        return self.prev_feat < other.prev_feat

    # 重载__str__和__repr__方法，同样是为了方便调试和可视化
    def __str__(self):
        if self.category is None:
            return "CvDNode ({}) ({} -> {})".format(
                self._depth, self.prev_feat, self.feature_dim)
        return "CvDNode ({}) ({} -> {})".format(
            self._depth, self.prev_feat, self.tree.label_dic[self.category])

    __repr__ = __str__

    # 定义children属性，主要是区分开连续+CART的情况和其余情况
    # 有了该属性以后，想要获得所有子节点时就不用分情况讨论了
    @property
    def children(self):
        return {
            "left": self.left_child,
            "right": self.right_child
        } if (self.is_cart or self.is_continuous) else self._children

    # 递归定义height属性
    # 叶节点高度都定义为1，其余节点的高度定义为最高的子节点的高度+1
    @property
    def height(self):
        if self.category is not None:
            return 1
        return 1 + max([
            _child.height if _child is not None else 0
            for _child in self.children.values()
        ])

    # 定义info_dic属性，它记录了该Node的主要信息
    # 在更新各个Node的叶节点时，被记录进各个self.leafs属性的就是该字典
    @property
    def info_dic(self):
        return {"chaos": self.chaos, "y": self._y}

    # p84
