import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# 1. 数据加载和预处理
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


# 2. 特征提取
def extract_features(data):
    """
    使用TF-IDF将文本转换为特征向量
    """
    tfidf = TfidfVectorizer(
        max_features=5000,  # 限制特征数量
        stop_words='english',  # 移除停用词
        lowercase=True,  # 转换为小写
        decode_error='ignore'  # 忽略解码错误
    )
    X = tfidf.fit_transform(data['text'])
    y = data['label'].values
    return X, y, tfidf


# 3. 模型训练和评估
def train_and_evaluate(X, y):
    """
    训练SVM模型并评估性能
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 创建SVM分类器
    clf = svm.SVC(
        kernel='linear',  # 线性核函数
        C=1.0,  # 正则化参数
        probability=False,  # 不需要概率估计
        class_weight='balanced'  # 处理类别不平衡
    )

    # 训练模型
    clf.fit(X_train, y_train)

    # 在测试集上预测
    y_pred = clf.predict(X_test)

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return clf


# 4. 预测新邮件
def predict_email(model, tfidf, email_text):
    """
    使用训练好的模型预测新邮件
    """
    # 将新邮件转换为特征向量
    features = tfidf.transform([email_text])
    # 预测
    prediction = model.predict(features)
    probability = model.decision_function(features)

    if prediction[0] == 1:
        print("This email is classified as SPAM.")
        print(f"Decision function value: {probability[0]:.4f} (higher values indicate stronger spam confidence)")
    else:
        print("This email is classified as HAM (not spam).")
        print(f"Decision function value: {probability[0]:.4f} (lower values indicate stronger ham confidence)")


# 主函数
def main():
    # 加载数据 (假设数据文件名为spam.csv)
    try:
        data = load_data('spam.csv')
    except FileNotFoundError:
        print("Error: File 'spam.csv' not found.")
        print("Please download the dataset from https://www.kaggle.com/uciml/sms-spam-collection-dataset")
        return

    print("Dataset loaded successfully.")
    print(f"Total emails: {len(data)}")
    print(f"Spam emails: {data['label'].sum()}")
    print(f"Ham emails: {len(data) - data['label'].sum()}")

    # 提取特征
    X, y, tfidf = extract_features(data)

    # 训练和评估模型
    print("\nTraining SVM model...")
    model = train_and_evaluate(X, y)

    # 示例预测
    print("\nExample predictions:")
    test_emails = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "Hi John, just wanted to check if we're still meeting tomorrow at 3pm.",
        "Congratulations! You've been selected for a special offer. Claim your prize now!",
        "Meeting reminder: Project review at 2pm in conference room B."
    ]

    for email in test_emails:
        print("\n" + "=" * 50)
        print(f"Email: {email}")
        predict_email(model, tfidf, email)


if __name__ == "__main__":
    main()