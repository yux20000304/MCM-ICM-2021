#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas_profiling
import seaborn as sns


# In[2]:


df_cost=pd.read_csv('./data/government-expenditure-on-tertiary-education-by-country.csv')
df_gdp_rate=pd.read_csv('./data/government-expenditure-per-student-tertiary-of-gdp-per-capita.csv')
df_enrollment=pd.read_csv('./data/gross-enrollment-ratio-in-tertiary-education.csv')
df_gpi=pd.read_csv('./data/gpi-tertiary-education.csv')
df_sex_rate=pd.read_csv('./data/percentage-of-all-students-in-tertiary-education-enrolled-in-isced-6-both-sexes.csv')
df_female=pd.read_csv('./data/share-graduates-stem-female.csv')
df_15_edu=pd.read_csv('./data/population-breakdown-by-highest-level-of-education-achieved-for-those-aged-15-in.csv')
df_15_edu_country=pd.read_csv('./data/projections-of-the-share-of-the-population-aged-15-educated-to-degree-level-by-country.csv')
df_school_life=pd.read_csv('./data/school-life-expectancy-from-primary-to-tertiary-education.csv')
df_private_edu=pd.read_csv('./data/precentage-enrolled-in-private-institutions-at-the-tertiary-education-level.csv')
df_from_abroad=pd.read_csv('./data/share-of-students-from-abroad.csv')
df_studying_abroad=pd.read_csv('./data/share-of-students-studying-abroad.csv')
df_post_secondary=pd.read_csv('./data/share-of-the-population-with-a-completed-post-secondary-education.csv')
df_complete_edu=pd.read_csv('./data/share-of-the-population-with-completed-tertiary-education.csv')
df_secondary_no_tertiary=pd.read_csv('./data/share-of-the-population-with-secondary-education-but-no-tertiary-education.csv')
df_gender_ratio=pd.read_csv('./data/gender-ratios-for-mean-years-of-schooling.csv')
df_female_average_eductaion=pd.read_csv('./data/mean-years-of-schooling-female.csv')
df_male_average_education=pd.read_csv('./data/mean-years-of-schooling-male.csv')


# ### 下面的几个函数分别代表了
# - 把最小型转换成最大型 dataDirection_1
# - 把中间型转换成最大型 dataDirection_2
# - 把区间型转换成最大型 dataDirection_3
# - 矩阵标准化 temp2
# - 计算得分之后进行归一化 temp3

# In[3]:


#min_scoring
def dataDirection_1(datas):         
        return np.max(datas)-datas     

#mid_scoring
def dataDirection_2(datas, x_best):
    temp_datas = datas - x_best
    M = np.max(abs(temp_datas))
    answer_datas = 1 - abs(datas - x_best) / M     
    return answer_datas
    
#period_scoring
def dataDirection_3(datas, x_min, x_max):
    M = max(x_min - np.min(datas), np.max(datas) - x_max)
    answer_list = []
    for i in datas:
        if(i < x_min):
            answer_list.append(1 - (x_min-i) /M)      
        elif( x_min <= i <= x_max):
            answer_list.append(1)
        else:
            answer_list.append(1 - (i - x_max)/M)
    return np.array(answer_list)   

#matrix_standard
def temp2(datas):
    K = np.power(np.sum(pow(datas,2),axis =1),0.5)
    for i in range(0,K.size):
        for j in range(0,datas[i].size):
            datas[i,j] = datas[i,j] / K[i]      
    return datas

#normalized
def temp3(answer2):
    list_max = np.array([np.max(answer2[0,:]),np.max(answer2[1,:]),np.max(answer2[2,:]),np.max(answer2[3,:])])  
    list_min = np.array([np.min(answer2[0,:]),np.min(answer2[1,:]),np.min(answer2[2,:]),np.min(answer2[3,:])])  
    max_list = []       
    min_list = []       
    answer_list=[]      
    for k in range(0,np.size(answer2,axis = 1)):        
        max_sum = 0
        min_sum = 0
        for q in range(0,4):                                
            max_sum += np.power(answer2[q,k]-list_max[q],2)
            min_sum += np.power(answer2[q,k]-list_min[q],2)     
        max_list.append(pow(max_sum,0.5))
        min_list.append(pow(min_sum,0.5))
        answer_list.append(min_list[k]/ (min_list[k] + max_list[k]))    
        max_sum = 0
        min_sum = 0
    answer = np.array(answer_list)      
    return (answer / np.sum(answer))


# ### 这里是第一个特征代表了高等学校人均政府支出占人均GDP的百分比
# - 这里我们把它当最小型处理，转换成最大型

# In[4]:


#GDP_rating
year=[]
country=['United States','Japan','Turkey',
'Poland','United Kingdom',
'Germany',
'France',
'Switzerland','Sweden','India']
average=[]
rate_score=[]

for i in range(2005,2015):
    year.append(str(i))
    
#calculate the average rate between 2005 and 2015
for temp in country:
    rate=df_gdp_rate[((df_gdp_rate["Entity"].isin([temp]))&(df_gdp_rate["Year"].isin(year)))]
    average.append(rate["Government expenditure per student, tertiary (% of GDP per capita)"].mean())
gdp_rate_score=dataDirection_1(average)
gdp_rate_score


# In[5]:


# average=[]
# contient_name=['Advanced Economies,Asia and the Pacific,Eastern Europe',
# 'Latin America and the Caribbean',
# 'Middle East and North Africa',
# 'Sub-Saharan Africa'
# ]
# for temp in country:
#     rate=df_gender_ratio[((df_gender_ratio["Entity"].isin([contient_name]))&(df_gender_ratio["Year"].isin(year)))]
#     average.append(rate["Regional female to male years schooling (Lee-Lee (2016))"].mean())
# average=np.array(average)


# ### 这里是第二个特征，GPI
# - 这个参数的含义是
# $$ GPI=女性净入学率/男性净入学率 $$
# - 我这里把它当作区间型来处理，主要是在worldbank上面对于这个性别均等的定义是GPI位于0.97和1.03之间

# In[6]:


#gpi_rating
average=[]
for temp in country:
    rate=df_gpi[((df_gpi["Entity"].isin([temp]))&(df_gpi["Year"].isin(year)))]
    average.append(rate["Gross enrolment ratio, tertiary, gender parity index (GPI)"].mean())
average=np.array(average)
gender_rate_score=dataDirection_3(average,0.97,1.03)
gender_rate_score


# ### 这里是第三个特征
# - 这个参数的含义是在该国高等教育总入学人数中来自国外学生的份额
# - 我们这里把它当作区间型进行处理，希望的区间是20-50（相关文献获得的）

# In[7]:


#students from abroad
average=[]
for temp in country:
    rate=df_from_abroad[((df_from_abroad["Entity"].isin([temp]))&(df_from_abroad["Year"].isin(year)))]
    average.append(rate["Inbound mobility rate, both sexes (%)"].mean())
average=np.array(average)
from_abroad_rate_score=dataDirection_3(average,20,50)
from_abroad_rate_score


# ### 这里是第四个特征
# - 这个参数的含义是给定国家在国外学习的学生人数占该国高等教育总入学人数的百分比
# - 这里我们把它当作最小型分布进行处理

# In[8]:


#students go abroad
average=[]
for temp in country:
    rate=df_studying_abroad[((df_studying_abroad["Entity"].isin([temp]))&(df_studying_abroad["Year"].isin(year)))]
    average.append(rate["Outbound mobility ratio, all regions, both sexes (%)"].mean())
average=np.array(average)
go_abroad_rate_score=dataDirection_1(average)
go_abroad_rate_score


# In[9]:


# #students with post secondary education
# average=[]
# for temp in country:
#     rate=df_post_secondary[((df_post_secondary["Entity"].isin([temp]))&(df_post_secondary["Year"].isin(year)))]
#     average.append(rate["UIS: Percentage of population age 25+ with at least completed post-secondary education (ISCED 4 or higher). Total"].mean())
# average=np.array(average)
# post_sceondary_rate_score=average
# post_sceondary_rate_score


# ### 这里是第五个特征
# - 这个参数代表了15岁以上该国人口接受了高等教育的人口比例
# - 这里我们把这个参数当作最大型，不需要进行处理

# In[10]:


#students with complete education
average=[]
for temp in country:
    rate=df_complete_edu[((df_complete_edu["Entity"].isin([temp]))&(df_complete_edu["Year"].isin(year)))]
    average.append(rate["Barro-Lee: Percentage of population age 15+ with tertiary schooling. Completed Tertiary"].mean())
average=np.array(average)
complete_edu_rate_score=average
complete_edu_rate_score


# ### 这里是我们的第六个特征
# - 这个特征的含义是接受私立大学高等教育的比例
# - 这里我们把它当作最大型进行处理
# - 注意我这里的翻译可能有点问题，因为英国的数据全是100，但是英国有很多公立大学
# - 原数据英文描述share enrolled in private institutions at the tertiary education level

# In[11]:


#students with private education
average=[]
for temp in country:
    rate=df_private_edu[((df_private_edu["Entity"].isin([temp]))&(df_private_edu["Year"].isin(year)))]
    average.append(rate["Percentage of enrolment in tertiary education in private institutions (%)"].mean())
average=np.array(average)
private_edu_rate_score=average
private_edu_rate_score


# ### 下面就是进行数据处理，对军阵进行操作最后获得归一化打分数组

# In[12]:


answer1=[gdp_rate_score,gender_rate_score,from_abroad_rate_score,go_abroad_rate_score,
         complete_edu_rate_score,private_edu_rate_score
        ]
answer1=np.array(answer1)
answer2=temp2(answer1)
answer3=temp3(answer2)
data=pd.DataFrame(answer3)

answer3
data


# ### 可视化

# In[14]:


x=[0,1,2,3,4,5,6,7,8,9]
plt.figure(figsize=(15, 8), dpi=800)
plt.bar(x,answer3,color='skyblue',tick_label = country,label='Accuary')
plt.ylabel('score',fontsize='18')
plt.xlabel('country',fontsize='18')
plt.title('Comparation between each country',fontsize='25')
plt.ylim(0,0.2)
plt.show()
sns.set(style='darkgrid')
plt.show()


# In[ ]:




