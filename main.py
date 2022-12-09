import random
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

#класс узла дерева
class CARTNode: #value: (objects, target, jiny, distribution); sign: (attr_idx, attr_val)
    def __init__(self, value : tuple, sign : tuple):
        self.value = value # для всех узлов дерева
        self.sign = sign   # для всех узлов дерева, кроме листьев
        self.left = None
        self.right = None

#оптимальный способ определения загрязнённости джини
def gini_impurity(target: np.ndarray, classes_quan : int):

    target = np.sort(target)
    target_size = target.shape[0]
    distr = []
    start_obj_ind = 0
    impurity = 1

    for cur_class in range(classes_quan):
        for obj_ind in range(start_obj_ind, target_size):

            if target[obj_ind] != cur_class:
                distr.append(obj_ind - start_obj_ind)
                start_obj_ind = obj_ind
                break

            if obj_ind == target_size - 1:
                distr.append(target_size - start_obj_ind)


    for dist in distr:
        impurity = impurity - (dist/target_size)**2

    return (impurity, distr)

#метод разделения данных (data: (objects, target))
def split_data(data, attr_idx, attr_val):

    data_list = list(data[0])
    data_target_list = list(data[1])
    left_list = []
    left_target_list = []

    right_list = []
    right_target_list = []

    for point_idx in range(len(data_list)):
        if data_list[point_idx][attr_idx] <= attr_val:
            left_list.append(data_list[point_idx])
            left_target_list.append(data_target_list[point_idx])
        else:
            right_list.append(data_list[point_idx])
            right_target_list.append(data_target_list[point_idx])

    left_data = (np.array(left_list), np.array(left_target_list))
    right_data = (np.array(right_list), np.array(right_target_list))

    return left_data, right_data

#метод разделения уззла дерева на 2 ветви
def split_node(root: CARTNode, attr_steps: int, classes_quan: int): #value: (objects, target, jiny, distribution)

    min_pair_impurity = 1.0

    for attr_idx in range(root.value[0].shape[1]):
        maxval = np.amax(root.value[0][:, attr_idx])
        minval = np.amin(root.value[0][:, attr_idx])
        step = (maxval-minval)/attr_steps

        for attr_val in range(1, attr_steps):
            val = round(minval + attr_val*step, 2)
            left_data, right_data = split_data((root.value[0], root.value[1]), attr_idx, val)
            left_impurity_and_dist = gini_impurity(left_data[1], classes_quan)
            right_impurity_and_dist = gini_impurity(right_data[1], classes_quan)

            cur_pair_impuity = (left_data[1].shape[0] * left_impurity_and_dist[0] + right_data[1].shape[0] * right_impurity_and_dist[0])/root.value[1].shape[0]
            if cur_pair_impuity < min_pair_impurity:

                min_pair_impurity = cur_pair_impuity

                best_left_data = left_data
                best_left_impurity_and_dist = left_impurity_and_dist

                best_right_data = right_data
                best_right_impurity_and_dist = right_impurity_and_dist

                best_attr_idx = attr_idx
                best_attr_val = val

    root.sign = (best_attr_idx, best_attr_val)
    root.left = CARTNode((best_left_data[0], best_left_data[1], round(best_left_impurity_and_dist[0], 4), best_left_impurity_and_dist[1]), ())
    root.right = CARTNode((best_right_data[0], best_right_data[1], round(best_right_impurity_and_dist[0], 4), best_right_impurity_and_dist[1]), ())
    
    return

#проверка, является ли данный узел листиком        
def is_leaf(node: CARTNode):
    is_leaf = False
    if node.sign == ():
        is_leaf = True
    return is_leaf

#обучение дерева по известному корню дерева
def CART_learn(root: CARTNode, levels : int, classes_quan, attr_step: float):
    if levels > 1:
        cur_node = root
        if cur_node.value[2] != 0:
            split_node(cur_node, attr_step, classes_quan)
            CART_learn(cur_node.left, levels - 1, classes_quan, attr_step)
            CART_learn(cur_node.right, levels - 1, classes_quan, attr_step)

#создание дерева с нуля (по датасету, максимальному количеству уровней, количеству класов, количеству шагов каждого атрубута)
def CART_create(dataset, tree_levels, classes_quantity, attr_steps):
    impurity_dist = gini_impurity(dataset[1], 3)
    tree_root = CARTNode((dataset[0], dataset[1], round(impurity_dist[0], 4), impurity_dist[1]), ()) #value is (objects, target, jiny, distribution)
    CART_learn(tree_root, tree_levels, classes_quantity, attr_steps)

    return tree_root

labels = ["sepal length in cm <=",
"sepal width in cm <=",
"petal length in cm <=",
"petal width in cm <=",
]


#прямой обход дерева
def CART_order(tree_root : CARTNode):

    cur_node = tree_root

    print('gini:\n', cur_node.value[2])
    print('distribution:\n', cur_node.value[3])
    
    if is_leaf(cur_node):
        print('Листик\n')
    else:
        print(labels[cur_node.sign[0]], cur_node.sign[1], '\n')


    if cur_node.left != None:
        CART_order(cur_node.left)

    if cur_node.right != None:
        CART_order(cur_node.right)

    return None

#выработать предсказание созданного дерева для одной точки множества
def CART_single_prediction(point, tree_node):

    if is_leaf(tree_node) == True:
        return tree_node.value[3].index(max(tree_node.value[3]))
    if point[tree_node.sign[0]] <= tree_node.sign[1]:
        return CART_single_prediction(point, tree_node.left)
    else:
        return CART_single_prediction(point, tree_node.right)

#выработать предсказания созданного дерева для всего набора данных
def CART_predict(dataset, tree_root):

    classes_lst = []
    for point in dataset:
        prediction = CART_single_prediction(point, tree_root)
        classes_lst.append(prediction)

    return classes_lst
    
#стратифицированное разделение набора данных ирисов на обучающий и тестовый
def custom_stratified_split(data_and_target: tuple, test_set_size, each_class):
    train_set_points = list(data_and_target[0])
    train_set_classes = list(data_and_target[1])
    test_set_points = []
    test_set_classes = []
    for i in range(test_set_size // 3):
        elem = train_set_points.pop(-1)
        test_set_points.append(elem)
        elem = train_set_classes.pop(-1)
        test_set_classes.append(elem)

        elem = train_set_points.pop(each_class*2 - 1)
        test_set_points.append(elem)
        elem = train_set_classes.pop(each_class*2 - 1)
        test_set_classes.append(elem)

        elem = train_set_points.pop(each_class - 1)
        test_set_points.append(elem)
        elem = train_set_classes.pop(each_class - 1)
        test_set_classes.append(elem)

        each_class = each_class - 1 

    return (np.array(train_set_points), np.array(train_set_classes)), (np.array(test_set_points), np.array(test_set_classes))



np.random.seed(10)
random.seed(10)

irises = datasets.load_iris(return_X_y=True)

attr_steps = 10 #количество сравнений на каждый атрибут
classes_quantity = 3 #количество классов в наборе данных
tree_levels = 4 #максимальное число уровней дерева
dataset = irises #набор данных



tree_root = CART_create(dataset, tree_levels, classes_quantity, attr_steps)
classes_lst = CART_predict(dataset[0], tree_root)
CART_order(tree_root)
print('Список классов:\n', classes_lst)

#разбиение набора данных для последующей проверки устойчивости дерева 
'''
train1, test1 = custom_stratified_split(dataset, 51, 50)
train2, useless = custom_stratified_split(train1, 15, 33)
train3, useless = custom_stratified_split(train2, 15, 24)
train4, useless = custom_stratified_split(train3, 15, 21)

train_sets_list = [train1, train2, train3, train4]
'''

#получение зависимости точности от количества уровней дерева
'''
accs = []
levels = []
for tree_levels in range(2, 9):
    print('max levels:\n', tree_levels)
    tree_root = CART_create(train1, tree_levels, classes_quantity, attr_steps)
    classes_lst = CART_predict(test1[0], tree_root)
    levels.append(tree_levels)
    accs.append(accuracy_score(classes_lst, test1[1]))

#CART_order(tree_root)
    print('accuracy:\n', accuracy_score(classes_lst, test1[1]))
    print('')

fig, ax = plt.subplots()
ax.plot(levels, accs)
ax.set_xlabel('levels')
ax.set_ylabel('accuracy')
plt.show()
'''


#получение зависимости точности от количества записей в обучающей выборке
'''
accs = []
train_lens = []
for train_set in train_sets_list:

    tree_root = CART_create(train_set, tree_levels, classes_quantity, attr_steps)
    classes_lst = CART_predict(test1[0], tree_root)
    accs.append(accuracy_score(classes_lst, test1[1]))
    train_lens.append(train_set[0].shape[0])
    print('train set length:\n', train_set[0].shape[0])
    print('accuracy:\n', accuracy_score(classes_lst, test1[1]))

fig, ax = plt.subplots()
ax.plot(train_lens, accs)
ax.set_xlabel('train set length')
ax.set_ylabel('accuracy')
plt.show()
'''