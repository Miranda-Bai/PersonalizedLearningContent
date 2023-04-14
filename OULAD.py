# Required packages
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import missingno as msno
from plotnine import *
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load Datasets
course = pd.read_csv('./OULAD/courses.csv')
assesment = pd.read_csv('./OULAD/assessments.csv')
vle = pd.read_csv('./OULAD/vle.csv')
info_stu = pd.read_csv('./OULAD/studentInfo.csv')
reg_stu = pd.read_csv('./OULAD/studentRegistration.csv')
as_stu = pd.read_csv('./OULAD/studentAssessment.csv')
vle_stu = pd.read_csv('./OULAD/studentVle.csv')

# Code module and code presentation are always used together to identify a module. 
# Therefore, these 2 column will be merged to create a feature which uniquely identifies a module.

course["module_presentation"] = list(map(lambda x,y: str(x) + "_" + str(y), course.code_module, course.code_presentation))
assesment["module_presentation"] = list(map(lambda x,y: str(x) + "_" + str(y), assesment.code_module, assesment.code_presentation))
vle["module_presentation"] = list(map(lambda x,y: str(x) + "_" + str(y), vle.code_module, vle.code_presentation))
info_stu["module_presentation"] = list(map(lambda x,y: str(x) + "_" + str(y), info_stu.code_module, info_stu.code_presentation))
reg_stu["module_presentation"] = list(map(lambda x,y: str(x) + "_" + str(y), reg_stu.code_module, reg_stu.code_presentation))
vle_stu["module_presentation"] = list(map(lambda x,y: str(x) + "_" + str(y), vle_stu.code_module, vle_stu.code_presentation))


print(" In Student Info table; \n # of rows: {} \n # of unique student Ids: {} \n \
Hence, There are different records for same students' different modul enrolments." .format(len(info_stu),len(info_stu.id_student.unique())))
print("\n This also means that student ID is not really a unique identifier of the table.\n")

# Missing value plot for vle 
msno.matrix(vle, figsize=(6,6), fontsize=15)
vle.drop(columns=['week_from','week_to'],inplace=True)
msno.bar(reg_stu, figsize=(6,6), fontsize=15)

#Find unregistered students according to registration table. 
#Then check whether they are consistent with the final results at StudentInfo table. 
#If a student is unregistered, final result must be recorded as "Withdrawn".


#Select unregistered students according to registration table
temp = reg_stu.loc[reg_stu.date_unregistration.notna(),\
                               ['id_student','module_presentation','date_unregistration']]

# Join to see matching rows
temp =pd.merge(info_stu, temp, on=['id_student','module_presentation'])

# Unregistered students without a "Withdrawn" in final result column 
# Semantic Error -- If a student unregistered, have to have "Withdrawn" as final result! 
wrong_final_results=temp.loc[temp.final_result!='Withdrawn']
incorrect_final_results=wrong_final_results.index
wrong_final_results

# Correction info_stu table's final_result entries
for i in wrong_final_results[['id_student','code_module','code_presentation']].values:
    info_stu.loc[(info_stu.id_student==i[0])&(info_stu.code_module==i[1])&\
                 (info_stu.code_presentation==i[2]),'final_result'] = 'Withdrawn'

assesment.groupby(['code_module','code_presentation']).agg(total_weight = ('weight',sum))

assesment[assesment.code_module.isin(["CCC","GGG"])]\
.groupby(['code_module','code_presentation',"assessment_type"]).agg(type_weights = ('weight',sum))

# Weights of exams are halved for module CCC
assesment.loc[(assesment.code_module=='CCC') &(assesment.assessment_type=='Exam'),'weight'] = \
assesment.loc[(assesment.code_module=='CCC') &(assesment.assessment_type=='Exam'),'weight']/2

# Weights of TMA type assessments arranged to be %100.
assesment.loc[(assesment.code_module=='GGG') & (assesment.assessment_type=='TMA'),'weight']=(100/3)

# Calculation of the marks by merging 2 tables to have assignment scores and weights together in one table.

# Join Assessment and StudentAssessment tables
joined=pd.merge(as_stu,assesment,on='id_assessment',how='left')
# Calculate weighted scores for all assessments of all students
joined['score*weight']=(joined['score']*joined['weight'])

# Sum up score*weights and divide by total weights (There are some students has total weight higher or much lower than %100)
# for all students of all modules to calculate final mark.
marks=joined.groupby(['id_student','code_module','code_presentation'],as_index=False)['score*weight','weight'].sum()

marks['adjusted_mark'] = marks['score*weight']/marks['weight']
marks["mark"]  = marks['score*weight']/200
marks.rename(columns = {'score*weight': 'total_score*weight', 'weight': 'attempted_weight'}, inplace=True)
marks = marks.round(1)

# Merging the marks table with info_stu to have a bigger table 
# containing all the relevant information about success, student characteristics and demographics.
joined = pd.merge(marks,info_stu,on=['id_student','code_module','code_presentation'],how='left')

# There can be students who attempt some of the assignments but then withdraw the course,
# mark variable may have a value for these students.
# These marks shouldn't be used in analysis so will be replaced with NaN as follows.
joined.loc[joined.final_result=='Withdrawn','mark']= np.nan
joined.loc[joined.final_result=='Withdrawn','adjusted_mark']= np.nan

ggplot(joined) + geom_boxplot(aes(x="final_result", y="attempted_weight"))

ggplot(joined[joined.final_result=="Distinction"]) + geom_point(aes(x="mark", y="adjusted_mark", color="attempted_weight"))\
+ggtitle("Students with Distinction Final Result")

ggplot(joined[joined.final_result=="Pass"]) + geom_point(aes(x="mark", y="adjusted_mark", color="attempted_weight"))\
+ggtitle("Students with Pass Final Result")

ggplot(joined[joined.final_result=="Fail"]) + geom_point(aes(x="mark", y="adjusted_mark", color="attempted_weight"))\
+ggtitle("Students with Fail Final Result")

distinction = joined[joined.final_result=="Distinction"].index
passing = joined[joined.final_result=="Pass"].index
fail = joined[joined.final_result=="Fail"].index
withdraw = joined[joined.final_result=="Withdraw"].index


attempt_weight200 = joined[joined.attempted_weight==200].index
attempt_weight150_200 = joined[(joined.attempted_weight>=150) & (joined.attempted_weight!=200)].index
attempt_weight0_150 = joined.index.difference(attempt_weight200).difference(attempt_weight150_200)

adj_mark80_100 = joined[joined.attempted_weight>=80].index
adj_mark70_80 = joined[(joined.attempted_weight>=70) & (joined.attempted_weight<80)].index
adj_mark0_70 = joined[joined.attempted_weight<70].index

mark40_100 = joined[joined.attempted_weight>=40].index
mark0_40 = joined[joined.attempted_weight<40].index

# Students with attempted_weight 200
joined.loc[(attempt_weight200) & (adj_mark80_100), "final_result"] = "Distinction"
joined.loc[(attempt_weight200) & (adj_mark70_80), "final_result"] = "Pass"
joined.loc[(attempt_weight200) & (adj_mark0_70), "final_result"] = "Fail"

# Students with attempted_weight between 150 and 200
joined.loc[(attempt_weight150_200) & (passing) & (adj_mark80_100), "final_result"] = "Distinction"
joined.loc[(attempt_weight150_200) & (fail) & (mark40_100), "final_result"] = "Pass"
joined.loc[(attempt_weight150_200) & (fail) & (mark40_100), "adjusted_mark"] = joined.loc[(attempt_weight150_200) & (fail) & (mark40_100), "mark"]

# Students with attempted_weight lower than 150 -- Ordering and reassigning adj_mark as explained above
joined.loc[joined.loc[(attempt_weight0_150) & (distinction)].mark.sort_values().index,"adjusted_mark"] = np.arange(70.0, 100.0, 30/len(joined.loc[(attempt_weight0_150) & (distinction)]))
joined.loc[joined.loc[(attempt_weight0_150) & (passing)].mark.sort_values().index,"adjusted_mark"] = np.arange(40.0, 70.0, 30/len(joined.loc[(attempt_weight0_150) & (passing)]))
joined.loc[joined.loc[(attempt_weight0_150) & (fail)].mark.sort_values().index,"adjusted_mark"] = np.arange(0.0, 40.0, 40/len(joined.loc[(attempt_weight0_150) & (fail)]))

df = joined

sns.distplot(df.loc[df.mark.notnull(),"mark"])
sns.distplot(df.loc[df.adjusted_mark.notnull(),"adjusted_mark"])

result_counts =  pd.DataFrame(df.final_result.value_counts()).reset_index()
result_counts = result_counts.rename(columns={"index": "final_result", "final_result":"counts"})

chart = sns.barplot(x="final_result", y="counts"  ,data=result_counts)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

education_counts =  pd.DataFrame(df.highest_education.value_counts()).reset_index()
education_counts = education_counts.rename(columns={"index": "highest_education", "highest_education":"counts"})

chart = sns.barplot(x="highest_education", y="counts"  ,data=education_counts)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

# Merge highest_education groups into 2 main groups
df["edu"] = list(map(lambda x: "HE or higher" if (x in ["HE Qualification", "Post Graduate Qualification"]) \
                else "Level or lower",df.highest_education))

df = df.drop(["total_score*weight","highest_education","module_presentation"], axis=1)

# Clicks for each id_site for all enrolments of all students
clicks = pd.merge(vle_stu, vle, on=["id_site"])
clicks = clicks.loc[:,["code_module_x","code_presentation_x","id_student","id_site","date","sum_click","activity_type"]]
clicks.rename(columns = {'code_module_x': 'code_module', 'code_presentation_x': 'code_presentation'}, inplace=True)

# Number of students for each module and presentation
number_of_students = df.groupby(["code_module","code_presentation"])\
    .agg(stu_count =("id_student", lambda x: x.nunique())).reset_index()

# Total number of clicks
grouped_clicks = clicks.groupby(["code_module","code_presentation","activity_type"])\
.agg(total_click = ("sum_click",sum)).reset_index()

# Clicks per person
grouped_clicks=pd.merge(grouped_clicks, number_of_students, on=["code_module","code_presentation"], how="left")
grouped_clicks["click_per_person"] = (grouped_clicks["total_click"]/grouped_clicks["stu_count"]).round(1)

# Draw a stacked bar chart
ggplot(grouped_clicks, aes(fill="activity_type", y="click_per_person", x="code_module")) + \
    geom_bar(position="stack", stat="identity")

# Final result counts for 2 different education level under "edu"
result_counts = pd.DataFrame(df.groupby(["edu"]).final_result.value_counts())\
    .rename(columns={"final_result":"counts"}).reset_index()

# Draw the pie charts
labels = list(result_counts.final_result.unique())
fracs1 = list(result_counts.iloc[4:,].counts)
fracs2 = list(result_counts.iloc[:4,].counts)

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(321)
ax1.pie(fracs1, radius=np.sqrt(sum(fracs1)/sum(fracs2)), autopct='%.0f%%')
ax1.set_title("Level or lower", fontdict={'fontweight':"bold"}, pad=80)


ax2 = fig.add_subplot(322)
ax2.pie(fracs2,  autopct='%.0f%%')
ax2.set_title("HE or higher", fontdict={'fontweight':"bold"})


fig.legend(labels=labels, title="Final Results",prop={'size': 14})
plt.show()

# Calculate mean adjusted_mark for each imd_band
df.groupby(["imd_band"]).agg(avg_adjusted_mark=("adjusted_mark",lambda x: x.mean()))

# Distribution of adjusted_mark for different age groups
sns.violinplot(x=df.age_band, y=df.adjusted_mark, order=["0-35", "35-55","55<="])

# Divide students into 2 groups as explained above
group1 = df[df.imd_band.isin(["90-100%","80-90%","70-80%","60-70%","50-60%"])\
   &(df.edu=="HE or higher")&(df.age_band.isin(["55<=","35-55"])) &(df.adjusted_mark.notna())]

group2 = df[(df.imd_band.isin(['30-40%','20-30%','10-20','40-50%','0-10%']))\
   &(df.edu=="Level or lower")&(df.age_band=="0-35")&(df.adjusted_mark.notna())]

print("Number of students in group1 (more likely to be succesfull): {} \nNumber of students in group2 (less likely to be succesfull): {}"\
     .format(len(group1),len(group2)))

# Draw distribution of adjusted_mark for both groups
fig = plt.figure(figsize=(10,6))
sns.distplot(group1.adjusted_mark)
sns.distplot(group2.adjusted_mark)

fig.legend(labels=['Group1','Group2'])
plt.show()

# list of tuples that uniquely identifies each element of group1 and group2
g1_key_tuple=list(zip(group1.code_module,group1.code_presentation,group1.id_student))
g2_key_tuple=list(zip(group2.code_module,group2.code_presentation,group2.id_student))

# Filter the virtual learning environment(VLE)-clikcs to keep only group1 and group2 members related
clicks_filtered = clicks[pd.Series(list(zip(clicks['code_module'],\
                                      clicks['code_presentation'],clicks["id_student"]))).isin((g1_key_tuple + g2_key_tuple))]



# Sum up number of clicks for activity types 
summed_clicks = clicks_filtered.groupby(["code_module","code_presentation","id_student","activity_type"])\
    .agg(total_clicks = ("sum_click",sum)).reset_index()


# Create columns for all activity types to store number of clicks as a feature for each student
featured = pd.get_dummies(summed_clicks, columns=["activity_type"], prefix=[""], prefix_sep=[""])
featured = featured.drop_duplicates(subset=['code_module',"code_presentation","id_student"], keep="first")



# assign the number of clicks of each activity to the corresponding column and row 
for i, enrolment in summed_clicks.groupby(["code_module","code_presentation","id_student"], sort=False):
    
    # i = tuple of ("code_module","code_presentation","id_student") for each group
    # enrolment is a DataFrame that contain only one element of groupby object -- one group -- one enrolment
    
    # Find the corresponding row of enrolment in 
    featured.loc[(featured.code_module==i[0])&(featured.code_presentation==i[1])&(featured.id_student==i[2])\
              ,list(enrolment.activity_type)] += list(enrolment.total_clicks)

    
# Calculate total clicks for each enrolment    
featured['total_clicks']=featured.iloc[:,4:].sum(axis=1)

# Join derived features with student demographics for group1
featured = pd.merge(featured, df, on=['code_module',"code_presentation","id_student"])


# Indexes belongs to group1 and group2
g1_index = featured[(pd.Series(list(zip(featured['code_module'],featured['code_presentation'],featured["id_student"])))\
                           .isin((g1_key_tuple)))].index
g2_index = featured.index.difference(g1_index)

# Add a new column to store assigned group
featured.loc[g1_index,'Group'] = "1"
featured.loc[g2_index,'Group'] = "2"

# Difference of correlation values between group1 and group2 on correlation values of adjusted mark with other variables
abs(featured[featured.Group=="1"].corr()["adjusted_mark"]\
    - featured[featured.Group=="2"].corr()["adjusted_mark"]).sort_values().tail(10)

# Create a figure
fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

# Add first subplot to figure
ax = fig.add_subplot(3, 2, 1)
ax1 = sns.regplot(x="resource", y="adjusted_mark" , data=featured[featured.Group=="1"], label="Group 1",ci=95, ax=ax)
ax1 = sns.regplot(x="resource", y="adjusted_mark" , data=featured[(featured.Group=="2")], label= "GRoup 2",ci=95, ax=ax)
ax1.set_xlim(1,600)
ax1.set_ylim(1,110)

# Add second subplot to figure
ax = fig.add_subplot(3, 2, 2)
ax2 = sns.regplot(x="dataplus", y="adjusted_mark" , data=featured[featured.Group=="1"], label="Group 1",ci=95, ax=ax)
ax2 = sns.regplot(x="dataplus", y="adjusted_mark" , data=featured[(featured.Group=="2")], label= "Group 2",ci=95, ax=ax)
ax2.set_xlim(1,100)
ax2.set_ylim(1,110)

# Add third subplot to figure
ax = fig.add_subplot(3, 2, 3)
ax3 = sns.regplot(x="questionnaire", y="adjusted_mark" , data=featured[featured.Group=="1"], label="Group 1",ci=95, ax=ax)
ax3 = sns.regplot(x="questionnaire", y="adjusted_mark" , data=featured[(featured.Group=="2")], label= "GRoup 2",ci=95, ax=ax)
ax3.set_xlim(1,60)
ax3.set_ylim(1,110)


# Add fourth subplot to figure
ax = fig.add_subplot(3, 2, 4)
ax4 = sns.regplot(x="oucontent", y="adjusted_mark" , data=featured[featured.Group=="1"], label="Group 1",ci=95, ax=ax)
ax4 = sns.regplot(x="oucontent", y="adjusted_mark" , data=featured[(featured.Group=="2")], label= "GRoup 2",ci=95, ax=ax)
ax4.set_xlim(1,8000)
ax4.set_ylim(1,110)

# Add fiveth subplot to figure
ax = fig.add_subplot(3, 2, 5)
ax5 = sns.regplot(x="ouwiki", y="adjusted_mark" , data=featured[featured.Group=="1"], label="Group 1",ci=95, ax=ax)
ax5 = sns.regplot(x="ouwiki", y="adjusted_mark" , data=featured[(featured.Group=="2")], label= "GRoup 2",ci=95, ax=ax)
ax5.set_xlim(1,1000)
ax5.set_ylim(1,110)

# Add sixth subplot to figure
ax = fig.add_subplot(3, 2, 6)
ax6 = sns.regplot(x="total_clicks", y="adjusted_mark" , data=featured[featured.Group=="1"], label="Group 1",ci=95, ax=ax)
ax6 = sns.regplot(x="total_clicks", y="adjusted_mark" , data=featured[(featured.Group=="2")], label= "GRoup 2",ci=95, ax=ax)
ax6.set_xlim(0,12000)
ax6.set_ylim(1,110)


plt.show()