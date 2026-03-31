import numpy as np
from collections import defaultdict
import pandas as pd


class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        """
        初始化朴素贝叶斯分类器
        :param alpha: 拉普拉斯平滑系数
        """
        self.alpha = alpha
        self.class_probs = {}  # 存储类别的先验概率 P(y)
        self.feature_probs = {}  # 存储特征的条件概率 P(x|y)
        self.classes = None  # 存储类别列表
        self.vocab = set()  # 存储词汇表

    def fit(self, X, y):
        """
        训练模型
        :param X: 文本数据列表，如 ["free money", "hello world"]
        :param y: 对应的标签列表，如 [1, 0]
        """
        # 计算类别的先验概率 P(y)
        self.classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        for cls, count in zip(self.classes, class_counts):
            self.class_probs[cls] = count / total_samples

        # 统计每个类别中每个词的出现次数
        word_counts = defaultdict(lambda: defaultdict(int))
        for cls in self.classes:
            word_counts[cls] = defaultdict(int)

        # 构建词汇表并统计词频
        for text, cls in zip(X, y):
            for word in text.split():
                self.vocab.add(word)
                word_counts[cls][word] += 1

        # 计算条件概率 P(word|cls) 使用拉普拉斯平滑
        self.feature_probs = {}
        for cls in self.classes:
            total_words_in_class = sum(word_counts[cls].values())
            vocab_size = len(self.vocab)
            self.feature_probs[cls] = {}

            for word in self.vocab:
                count = word_counts[cls].get(word, 0) + self.alpha
                denominator = total_words_in_class + self.alpha * vocab_size
                self.feature_probs[cls][word] = count / denominator

    def predict(self, X):
        """
        预测新样本的类别
        :param X: 待预测的文本列表
        :return: 预测的类别列表
        """
        predictions = []
        for text in X:
            max_log_prob = -np.inf
            best_class = None

            # 对每个类别计算联合概率的对数
            for cls in self.classes:
                log_prob = np.log(self.class_probs[cls])  # log P(y)

                # 计算所有特征的对数概率和
                for word in text.split():
                    if word in self.feature_probs[cls]:
                        log_prob += np.log(self.feature_probs[cls][word])
                    # 如果词不在词汇表中，可以忽略或赋予一个很小的概率
                    # 这里我们选择忽略未登录词

                # 选择概率最大的类别
                if log_prob > max_log_prob:
                    max_log_prob = log_prob
                    best_class = cls

            predictions.append(best_class)
        return predictions

    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        :param X_test: 测试集文本
        :param y_test: 测试集真实标签
        """
        predictions = self.predict(X_test)
        accuracy = np.mean(np.array(predictions) == np.array(y_test))

        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        self._print_classification_report(predictions, y_test)
        print("\nConfusion Matrix:")
        self._print_confusion_matrix(predictions, y_test)

    def _print_classification_report(self, y_pred, y_true):
        """
        打印分类报告
        """
        classes = np.unique(y_true)
        for cls in classes:
            tp = np.sum((np.array(y_pred) == cls) & (np.array(y_true) == cls))
            fp = np.sum((np.array(y_pred) == cls) & (np.array(y_true) != cls))
            fn = np.sum((np.array(y_pred) != cls) & (np.array(y_true) == cls))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"Class {cls}:")
            print(f"  Precision: {precision:.2f}")
            print(f"  Recall: {recall:.2f}")
            print(f"  F1-score: {f1:.2f}")

    def _print_confusion_matrix(self, y_pred, y_true):
        """
        打印混淆矩阵
        """
        classes = np.unique(y_true)
        n_classes = len(classes)
        matrix = np.zeros((n_classes, n_classes), dtype=int)

        cls_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for true, pred in zip(y_true, y_pred):
            matrix[cls_to_idx[true]][cls_to_idx[pred]] += 1

        print("Rows represent true classes, columns represent predicted classes")
        print(" " * 5 + " ".join(f"{cls:^5}" for cls in classes))
        for i, cls in enumerate(classes):
            print(f"{cls:^5}" + " ".join(f"{num:^5}" for num in matrix[i]))

def load_data(filename):
    """
    加载数据集
    """
    data = pd.read_csv(filename, encoding='latin-1')
    # 重命名列名
    data = data[['v1', 'v2']]
    data.columns = ['label', 'text']
    # 将标签转换为二进制 (0: ham, 1: spam)
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data
data = load_data('spam.csv')
# 示例数据集


# 转换为DataFrame
df = pd.DataFrame(data)
X = df['text']
y = df['label']

# 划分训练集和测试集
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 训练模型
model = NaiveBayesClassifier(alpha=1.0)
model.fit(X_train, y_train)

# 评估模型
print("Model Evaluation on Test Set:")
model.evaluate(X_test, y_test)

# 预测新样本
new_emails = [
    'free meeting',  # 可能被分类为spam
    'project update',  # 可能被分类为ham
    'win lottery'  # 可能被分类为spam
]
predictions = model.predict(new_emails)
print("\nPredictions for new emails:")
for email, pred in zip(new_emails, predictions):
    print(f"'{email}' -> {'spam' if pred == 1 else 'ham'}")