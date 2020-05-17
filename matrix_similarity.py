import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def mtx_similar1(arr1:np.ndarray, arr2:np.ndarray) ->float:
    '''
    计算矩阵相似度的一种方法。将矩阵展平成向量，计算向量的乘积除以模长。
    注意有展平操作。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:实际是夹角的余弦值，ret = (cos+1)/2
    '''
    farr1 = arr1.ravel()
    farr2 = arr2.ravel()
    len1 = len(farr1)
    len2 = len(farr2)
    if len1 > len2:
        farr1 = farr1[:len2]
    else:
        farr2 = farr2[:len1]

    numer = np.sum(farr1 * farr2)
    denom = np.sqrt(np.sum(farr1**2) * np.sum(farr2**2))
    similar = numer / denom # 这实际是夹角的余弦值
    return  (similar+1) / 2     # 姑且把余弦函数当线性

def mtx_similar2(arr1:np.ndarray, arr2:np.ndarray) ->float:
    '''
    计算对矩阵1的相似度。相减之后对元素取平方再求和。因为如果越相似那么为0的会越多。
    如果矩阵大小不一样会在左上角对齐，截取二者最小的相交范围。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:相似度（0~1之间）
    '''
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0],arr2.shape[0])
        miny = min(arr1.shape[1],arr2.shape[1])
        differ = arr1[:minx,:miny] - arr2[:minx,:miny]
    else:
        differ = arr1 - arr2
    numera = np.sum(differ**2)
    denom = np.sum(arr1**2)
    similar = 1 - (numera / denom)
    return similar


def mtx_similar3(arr1:np.ndarray, arr2:np.ndarray) ->float:
    '''
    From CS231n: There are many ways to decide whether
    two matrices are similar; one of the simplest is the Frobenius norm. In case
    you haven't seen it before, the Frobenius norm of two matrices is the square
    root of the squared sum of differences of all elements; in other words, reshape
    the matrices into vectors and compute the Euclidean distance between them.
    difference = np.linalg.norm(dists - dists_one, ord='fro')
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:相似度（0~1之间）
    '''
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0],arr2.shape[0])
        miny = min(arr1.shape[1],arr2.shape[1])
        differ = arr1[:minx,:miny] - arr2[:minx,:miny]
    else:
        differ = arr1 - arr2
    dist = np.linalg.norm(differ, ord='fro')
    len1 = np.linalg.norm(arr1)
    len2 = np.linalg.norm(arr2)     # 普通模长
    denom = (len1 + len2) / 2
    similar = 1 - (dist / denom)
    return similar


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


'''calculate similarity'''
data = pd.read_csv('HFpEF data/aim3data__630NoMedicationstate_simple_nomerge.csv')
data_ori = data[data['Action'] == 1]
data_pre = data[data['top1_action90'] == 8]
data_ori = data_ori[['Gender', 'Age', 'Height', 'BW', 'BMI', 'BSA', 'SBP', 'DBP',
       'Pulse', 'O2_saturation', 'AMI', 'CVD', 'Cardiomyopathy', 'DM', 'HT',
       'Obesity', 'Chronic_Kidney_Disease', 'Arrhythmia (broad Rapid)',
       'Arrhythmia (broad Slow)', 'Atrial_Fibrillation',
       'Pulmonary_Hypertension', 'Anemia', 'Metabolic syndrome', 'Neoplasm',
       'Chemotherapy', 'NT-proBN', 'BNP', 'EGFR', 'CRP_HS', 'ESR', 'TROPONI_I',
       'TROPONIN_T', 'CK_MB', 'ALT', 'AST', 'BUN', 'PROTEIN', 'HDL', 'LDL',
       'WBC', 'HGB', 'PLT', 'RDW', 'pH', 'PO2', 'PCO2', 'FIO2', 'LVEDV',
       'LVEDVI', 'LVESV', 'LVESVI', 'LVEDD', 'LVESD', 'AS', 'PL', 'LVEF',
       'LVSV', 'CO', 'LVMASS', 'LVMASSI', 'RVEDV', 'RVEDVI', 'RVESV', 'RVESVI',
       'RVSV', 'RVEF', 'LA_Anteroposterior_Dimension',
       'RestRegionalWallMotion', 'RestPerfusion', 'LVLateEnhancemen',
       'Diuretics', 'MRA', 'PDE5I', 'Nitrate_etc', 'PKG-Stimulating_drugs',
       'Inotropics', 'Statins', 'Beta_blocker', 'ARB', 'ACEI', 'CCB',
       'Radiofrequency_catheter_ablation_for_atrial_fibrillation',
       'Cardioversion', 'PCI', 'CABG', 'HFpEF']]
data_pre = data_pre[['Gender', 'Age', 'Height', 'BW', 'BMI', 'BSA', 'SBP', 'DBP',
       'Pulse', 'O2_saturation', 'AMI', 'CVD', 'Cardiomyopathy', 'DM', 'HT',
       'Obesity', 'Chronic_Kidney_Disease', 'Arrhythmia (broad Rapid)',
       'Arrhythmia (broad Slow)', 'Atrial_Fibrillation',
       'Pulmonary_Hypertension', 'Anemia', 'Metabolic syndrome', 'Neoplasm',
       'Chemotherapy', 'NT-proBN', 'BNP', 'EGFR', 'CRP_HS', 'ESR', 'TROPONI_I',
       'TROPONIN_T', 'CK_MB', 'ALT', 'AST', 'BUN', 'PROTEIN', 'HDL', 'LDL',
       'WBC', 'HGB', 'PLT', 'RDW', 'pH', 'PO2', 'PCO2', 'FIO2', 'LVEDV',
       'LVEDVI', 'LVESV', 'LVESVI', 'LVEDD', 'LVESD', 'AS', 'PL', 'LVEF',
       'LVSV', 'CO', 'LVMASS', 'LVMASSI', 'RVEDV', 'RVEDVI', 'RVESV', 'RVESVI',
       'RVSV', 'RVEF', 'LA_Anteroposterior_Dimension',
       'RestRegionalWallMotion', 'RestPerfusion', 'LVLateEnhancemen',
       'Diuretics', 'MRA', 'PDE5I', 'Nitrate_etc', 'PKG-Stimulating_drugs',
       'Inotropics', 'Statins', 'Beta_blocker', 'ARB', 'ACEI', 'CCB',
       'Radiofrequency_catheter_ablation_for_atrial_fibrillation',
       'Cardioversion', 'PCI', 'CABG', 'HFpEF']]
cor_ori = np.cov(data_ori, rowvar=0)
cor_pre = np.cov(data_pre, rowvar=0)

print('method 1:')
print(mtx_similar1(cor_ori, cor_pre))
print('method 2:')
print(mtx_similar2(cor_ori, cor_pre))

'''PCA method'''
meanval_ori = np.mean(cor_ori, axis=0) #计算原始数据中每一列的均值，axis=0按列取均值
newData_ori = cor_ori-meanval_ori #去均值化，每个feature的均值为0
covMat_ori = np.cov(newData_ori, rowvar=0) #计算协方差矩阵，rowvar=0表示数据的每一列代表一个feature
featValue_ori, featVec_ori = np.linalg.eig(covMat_ori) #计算协方差矩阵的特征值和特征向量
index_ori = np.argsort(-featValue_ori) #将特征值按从大到小排序，index保留的是对应原featValue中的下标


meanval_pre = np.mean(cor_pre, axis=0) #计算原始数据中每一列的均值，axis=0按列取均值
newData_pre = cor_pre-meanval_pre #去均值化，每个feature的均值为0
covMat_pre = np.cov(newData_pre, rowvar=0) #计算协方差矩阵，rowvar=0表示数据的每一列代表一个feature
featValue_pre, featVec_pre = np.linalg.eig(covMat_pre) #计算协方差矩阵的特征值和特征向量
index_pre = np.argsort(-featValue_pre) #将特征值按从大到小排序，index保留的是对应原featValue中的下标

print('method 3:')
print(cos_sim(featVec_ori[:, 0], featVec_pre[:, 0]))


'''visualization, use the result of method 3'''
data = pd.read_csv('HFpEF data/aim3data__630NoMedicationstate_simple_nomerge.csv')
res = []
data_ori = data[data['Action'] == 1]
data_ori = data_ori[['Gender', 'Age', 'Height', 'BW', 'BMI', 'BSA', 'SBP', 'DBP',
       'Pulse', 'O2_saturation', 'AMI', 'CVD', 'Cardiomyopathy', 'DM', 'HT',
       'Obesity', 'Chronic_Kidney_Disease', 'Arrhythmia (broad Rapid)',
       'Arrhythmia (broad Slow)', 'Atrial_Fibrillation',
       'Pulmonary_Hypertension', 'Anemia', 'Metabolic syndrome', 'Neoplasm',
       'Chemotherapy', 'NT-proBN', 'BNP', 'EGFR', 'CRP_HS', 'ESR', 'TROPONI_I',
       'TROPONIN_T', 'CK_MB', 'ALT', 'AST', 'BUN', 'PROTEIN', 'HDL', 'LDL',
       'WBC', 'HGB', 'PLT', 'RDW', 'pH', 'PO2', 'PCO2', 'FIO2', 'LVEDV',
       'LVEDVI', 'LVESV', 'LVESVI', 'LVEDD', 'LVESD', 'AS', 'PL', 'LVEF',
       'LVSV', 'CO', 'LVMASS', 'LVMASSI', 'RVEDV', 'RVEDVI', 'RVESV', 'RVESVI',
       'RVSV', 'RVEF', 'LA_Anteroposterior_Dimension',
       'RestRegionalWallMotion', 'RestPerfusion', 'LVLateEnhancemen',
       'Diuretics', 'MRA', 'PDE5I', 'Nitrate_etc', 'PKG-Stimulating_drugs',
       'Inotropics', 'Statins', 'Beta_blocker', 'ARB', 'ACEI', 'CCB',
       'Radiofrequency_catheter_ablation_for_atrial_fibrillation',
       'Cardioversion', 'PCI', 'CABG', 'HFpEF']]
cor_ori = np.cov(data_ori, rowvar=0)
meanval_ori = np.mean(cor_ori, axis=0) #计算原始数据中每一列的均值，axis=0按列取均值
newData_ori = cor_ori-meanval_ori #去均值化，每个feature的均值为0
covMat_ori = np.cov(newData_ori, rowvar=0) #计算协方差矩阵，rowvar=0表示数据的每一列代表一个feature
featValue_ori, featVec_ori = np.linalg.eig(covMat_ori) #计算协方差矩阵的特征值和特征向量
index_ori = np.argsort(-featValue_ori) #将特征值按从大到小排序，index保留的是对应原featValue中的下标
res.append(list(featVec_ori[:, index_ori[0]]))

action = [1,2,3,6,8]
for a in action:
    data_pre = data[data['top1_action90'] == a]
    data_pre = data_pre[['Gender', 'Age', 'Height', 'BW', 'BMI', 'BSA', 'SBP', 'DBP',
           'Pulse', 'O2_saturation', 'AMI', 'CVD', 'Cardiomyopathy', 'DM', 'HT',
           'Obesity', 'Chronic_Kidney_Disease', 'Arrhythmia (broad Rapid)',
           'Arrhythmia (broad Slow)', 'Atrial_Fibrillation',
           'Pulmonary_Hypertension', 'Anemia', 'Metabolic syndrome', 'Neoplasm',
           'Chemotherapy', 'NT-proBN', 'BNP', 'EGFR', 'CRP_HS', 'ESR', 'TROPONI_I',
           'TROPONIN_T', 'CK_MB', 'ALT', 'AST', 'BUN', 'PROTEIN', 'HDL', 'LDL',
           'WBC', 'HGB', 'PLT', 'RDW', 'pH', 'PO2', 'PCO2', 'FIO2', 'LVEDV',
           'LVEDVI', 'LVESV', 'LVESVI', 'LVEDD', 'LVESD', 'AS', 'PL', 'LVEF',
           'LVSV', 'CO', 'LVMASS', 'LVMASSI', 'RVEDV', 'RVEDVI', 'RVESV', 'RVESVI',
           'RVSV', 'RVEF', 'LA_Anteroposterior_Dimension',
           'RestRegionalWallMotion', 'RestPerfusion', 'LVLateEnhancemen',
           'Diuretics', 'MRA', 'PDE5I', 'Nitrate_etc', 'PKG-Stimulating_drugs',
           'Inotropics', 'Statins', 'Beta_blocker', 'ARB', 'ACEI', 'CCB',
           'Radiofrequency_catheter_ablation_for_atrial_fibrillation',
           'Cardioversion', 'PCI', 'CABG', 'HFpEF']]
    cor_pre = np.cov(data_pre, rowvar=0)
    meanval_pre = np.mean(cor_pre, axis=0) #计算原始数据中每一列的均值，axis=0按列取均值
    newData_pre = cor_pre-meanval_pre #去均值化，每个feature的均值为0
    covMat_pre = np.cov(newData_pre, rowvar=0) #计算协方差矩阵，rowvar=0表示数据的每一列代表一个feature
    featValue_pre, featVec_pre = np.linalg.eig(covMat_pre) #计算协方差矩阵的特征值和特征向量
    index_pre = np.argsort(-featValue_pre) #将特征值按从大到小排序，index保留的是对应原featValue中的下标
    res.append(list(featVec_pre[:, index_pre[0]]))


pca=PCA(n_components=2)
reduced_res=pca.fit_transform(res)

zero = [0, 0]
text = ['doctor_1', '1', '2', '3', '6', '8']
for i in range(len(action)+1):
    plt.plot([reduced_res[i][0], 0], [reduced_res[i][1], 0], color='r')
    # plt.scatter([reduced_res[i][0], 0], [reduced_res[i][1], 0], color='r')
    plt.text(reduced_res[i][0]/2, reduced_res[i][1]/2, text[i], fontdict={'size':'10','color':'b'})
plt.show()
plt.savefig('action_pic/similarity.png')
