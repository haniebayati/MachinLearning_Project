#  کتابخانه های موردنیاز را import می کنیم
import pandas as pd
import statistics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# یک ستون را از اکسل خوانده و به عنوان ارایه برمیگردانیم
def read_excel_column(fileName, columnName):
        # فایل اکسل را میخوانیم
        data = pd.read_excel(fileName)

        # چک می کنیم که ستون در اکسل وجود دارد یا نه 
        if columnName not in data.columns:
            raise ValueError(f"Column '{columnName}' not found in the Excel file.")

        # مقادیر ستون را می گیریم
        column_data = data[columnName].astype(str) # اطمینان حاصل کنید که ستون به عنوان رشته در نظر گرفته می شود  
        return column_data

#  یک متد برای محاسبه knn تعریف می کنیم
def calculate_knn(final_data):
    
    # ستون های 2 تا 36 را برای ویژگی و ستون زبان را برای لیبل انتخاب می کنیم
    features = final_data.iloc[:, 2:35].values
    labels = final_data.iloc[:, 1].values 


    test_rows = [i for i in range(9, len(final_data), 10)]  # سطرهای 10، 20، ... به عنوان نمونه‌های تست انتخاب می شوند
    train_rows = [i for i in range(len(final_data)) if i not in test_rows]  #  بقیه سطرها به عنوان داده‌های آموزشی انتخاب می شوند
    
    # داده های تست و آموزش را می گیریم
    x_train = features[train_rows]
    x_test = features[test_rows]
    
    # برچسب های تست و آموزش را می گیریم
    y_train = labels[train_rows]
    y_test = labels[test_rows]

    #  x_train را با متد StandardScaler نرمال  می کنیم 
    X_train_norm = preprocessing.StandardScaler().fit(x_train).transform(x_train.astype(float)) # داده ها را float می کنیم
    #  x_test را با متد StandardScaler نرمال می کنیم
    X_test_norm = preprocessing.StandardScaler().fit(x_test).transform(x_test.astype(float)) # داده ها را float می کنیم
    
    
        # تعریف مدل KNN با k های مختلف
    for k in [1, 3, 5]:
        for metric in ['cosine', 'euclidean', 'minkowski']:
            # ایجاد مدل KNN
            knn_model = KNeighborsClassifier(n_neighbors=k, metric=metric)

            # آموزش مدل با داده‌های آموزشی
            knn_model.fit(X_train_norm, y_train)

            # پیش‌بینی برچسب‌ها برای داده‌های تست
            y_pred = knn_model.predict(X_test_norm)

            # محاسبه دقت  و خطای مدل
            accuracy = accuracy_score(y_test, y_pred)
            error = 1 - accuracy
            print(f'KNN with k={k}, metric={metric}: Accuracy = {accuracy:.2f} , Error = {error:.2f}')

    
# تعریف نام فایل و ستون موردنظر و فراخوانی تابع read_excel_column
fileName = 'HW1_USC_ML_4021.xlsx'
columnName = 'جملات' 
excel_column = read_excel_column(fileName, columnName)


# فقط در صورتی ادامه می دهیم که exel_column خالی نباشد
if excel_column is not None:
    # ایجاد آرایه الفبای ترکیبی (ترکیب الفبای فارسی و عربی)
    arabic_alphabet = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ',
                       'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي' , 'ة', 'ء']
    persian_alphabet = ['ا', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط',
                        'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و', 'ه', 'ی']

    combined_alphabet = list(set(arabic_alphabet + persian_alphabet))
    combined_alphabet.sort() # مرتب سازی آرایه الفبای ترکیبی
    
    #  دیتافریم ها را می سازیم
    results = []
    norm_count = []   
    combined_alphabet_count_list = []
    z_norm_count = []
    #  برای هر سطر ستون انجام می دهیم
    for row in excel_column:
        # وجود یا عدم وجود هر کاراکتر را با 0 و 1 مشخص می کنیم
        char_presence = [1 if char in row else 0 for char in combined_alphabet]
        #  تعداد کاراکتر های هر سطر را می شماریم 
        count = sum(1 for char in row if char in combined_alphabet)

        #  دیکشنری ها را می سازیم
        count_dict = {}
        normal_count = {}
        z_normal_count = {}
        
        # برای هر کاراکتر در هر سطر انجام می دهیم
        for char in combined_alphabet:
            #  تعداد هر کاراکتر را می شماریم
            count_dict[char] = row.count(char)
            #  تعداد هر کاراکتر را به تعداد کاراکترهای سطر تقسیم کرده و نرمال شده آن را به دست می آوریم
            normal_count[char] = count_dict[char] / count
        #  میانگین و انحراف معیار را برای هر سطر به دست می آوریم    
        mean = statistics.mean (count_dict.values()) #  مقادیر موجود در دیکشنری را میانگین می گیریم
        stdv = statistics.stdev (count_dict.values()) #  مقادیر موجود در دیکشنری را میانگین می گیریم
        
        #  برای هر کاراکتر در هر سطر انجام می دهیم
        for char in combined_alphabet:
            #  فرمول z-score را محاسبه می کنیم
            z_normal_count[char] = (count_dict[char] - mean) / stdv 
        
        #  z-score کاراکترها را به دیتافریم اضافه می کنیم    
        z_norm_count.append(z_normal_count)  
        #  وزن کاراکترها را به دیتافریم اضافه می کنیم  
        combined_alphabet_count_list.append(count_dict)  
        #  نرمال شده کاراکترها را به دیتافریم اضافه می کنیم
        norm_count.append(normal_count)
        # وجود یا عدم وجود کاراکترها را به دیتافریم اضافه می کنیم
        results.append(char_presence)
    

    # ساخت یک دیتا فریم برای results
    results1 = pd.DataFrame(results, columns=combined_alphabet)
    # فایل اکسل اصلی را دوباره بخوانید تا نتایج به آن اضافه شود
    data1 = pd.read_excel(fileName)
    # داده های اصلی را با نتایج الحاق کنید (از ستون دوم شروع کنید)
    final_data1 = pd.concat ([data1, results1] , axis=1) 
    print('B_bow:')
    #  نتیجه را به متد محاسبه knn می دهیم
    calculate_knn(final_data1)
    
     # ساخت یک دیتا فریم برای combined_alphabet_count_list
    results2 = pd.DataFrame(combined_alphabet_count_list, columns=combined_alphabet)
    # فایل اکسل اصلی را دوباره بخوانید تا نتایج به آن اضافه شود
    data2 = pd.read_excel(fileName)
     # داده های اصلی را با نتایج الحاق کنید (از ستون دوم شروع کنید)
    final_data2 = pd.concat([data2, results2] , axis=1) 
    print('W_bow:')
    #  نتیجه را به متد محاسبه knn می دهیم
    calculate_knn(final_data2)
    
      # ساخت یک دیتا فریم برای norm_count
    results3 = pd.DataFrame(norm_count, columns=combined_alphabet)
    # فایل اکسل اصلی را دوباره بخوانید تا نتایج به آن اضافه شود
    data3 = pd.read_excel(fileName)
     # داده های اصلی را با نتایج الحاق کنید (از ستون دوم شروع کنید)
    final_data3 = pd.concat([data3, results3], axis=1)
    print('N_bow:')
    #  نتیجه را به متد محاسبه knn می دهیم
    calculate_knn(final_data3)

      # ساخت یک دیتا فریم برای z_norm_count
    results4 = pd.DataFrame(z_norm_count, columns=combined_alphabet)
    # فایل اکسل اصلی را دوباره بخوانید تا نتایج به آن اضافه شود
    data4 = pd.read_excel(fileName)
     # داده های اصلی را با نتایج الحاق کنید (از ستون دوم شروع کنید)
    final_data4 = pd.concat([data4, results4], axis=1)
    print('Z_bow:')
    #  نتیجه را به متد محاسبه knn می دهیم
    calculate_knn(final_data4)


