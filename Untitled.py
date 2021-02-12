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

#visualize
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