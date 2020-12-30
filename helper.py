from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Function to compute the metrics
def compute_metrics(train_pred, y_train, test_pred, y_test):
    # Confusion matrix for train predictions
    confmat = confusion_matrix(y_train, train_pred)
    
    print('Train metrics')
    print('Confusion matrix')
    print(confmat)
    print("----------------------------------------------")    
    TP = confmat[0,0]
    TN = confmat[1,1]
    FN = confmat[0,1]
    FP = confmat[1,0]
    Total = TP + TN + FP + FN

    # Accuracy: Overall, how often is the classifier correct?
    Accuracy = (TP+TN)/Total 
    # Misclassification Rate: Overall, how often is it wrong?
    # equivalent to 1 minus Accuracy also known as "Error Rate"
    Misclassification_Rate = (FP+FN)/Total

    # True Positive Rate: When it's actually yes, how often does it predict yes?
    # also known as "Sensitivity" or "Recall"
    Actual_Yes = TP + FN
    Recall = TP/Actual_Yes

    # False Positive Rate: When it's actually no, how often does it predict yes?
    Actual_No = TN + FP
    FPR = FP/Actual_No

    # True Negative Rate: When it's actually no, how often does it predict no?
    # equivalent to 1 minus False Positive Rate, also known as "Specificity"
    TNR = TN/Actual_No

    # Precision: When it predicts yes, how often is it correct?
    Predicted_Yes = TP + FP
    Precission = TP/Predicted_Yes

    # Prevalence: How often does the yes condition actually occur in our sample?
    Prevalance = Actual_Yes / Total
    
    # F1 Score
    f1 = 2 * (Precission * Recall) / (Precission + Recall)

    print('Accuracy: ', Accuracy)
    print('Precission: ', Precission)
    print('Recall: ', Recall)
    print('F1 Score: ', f1)
    print("")
    print("==============================================")
    print("")
    # Confusion matrix for train predictions
    confmat = confusion_matrix(y_test, test_pred)
    print('Test metrics')
    print('Confusion matrix')
    print(confmat)
    print("----------------------------------------------")  
    TP = confmat[0,0]
    TN = confmat[1,1]
    FN = confmat[0,1]
    FP = confmat[1,0]
    Total = TP + TN + FP + FN

    # Accuracy: Overall, how often is the classifier correct?
    Accuracy = (TP+TN)/Total 
    # Misclassification Rate: Overall, how often is it wrong?
    # equivalent to 1 minus Accuracy also known as "Error Rate"
    Misclassification_Rate = (FP+FN)/Total

    # True Positive Rate: When it's actually yes, how often does it predict yes?
    # also known as "Sensitivity" or "Recall"
    Actual_Yes = TP + FN
    Recall = TP/Actual_Yes

    # False Positive Rate: When it's actually no, how often does it predict yes?
    Actual_No = TN + FP
    FPR = FP/Actual_No

    # True Negative Rate: When it's actually no, how often does it predict no?
    # equivalent to 1 minus False Positive Rate, also known as "Specificity"
    TNR = TN/Actual_No

    # Precision: When it predicts yes, how often is it correct?
    Predicted_Yes = TP + FP
    Precission = TP/Predicted_Yes

    # Prevalence: How often does the yes condition actually occur in our sample?
    Prevalance = Actual_Yes / Total
    
    # F1 Score
    f1 = 2 * (Precission * Recall) / (Precission + Recall)

    print('Accuracy: ', Accuracy)
    print('Precission: ', Precission)
    print('Recall: ', Recall)
    print('F1 Score: ', f1)

# Function to draw plot for the train and validation accuracies
def accuracy_plot(history):
    plt.clf() # Clears the figure
    history_dict = history.history
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, acc_values, label='Training accuracy')
    plt.plot(epochs, val_acc_values, label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Function to draw plot for the train and validation loss 
def loss_plot(history):
    plt.clf() # Clears the figure
    history_dict = history.history
    acc_values = history_dict['loss']
    val_acc_values = history_dict['val_loss']
    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, acc_values, label='Training loss')
    plt.plot(epochs, val_acc_values, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()