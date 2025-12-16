load("C:/Users/aethe/Desktop/coursebook/Course materials/EB/Georgescu.Wren.RData")

library(tidyverse)
library(dplyr)
complete=complete[complete$mistake==0,]


```{r}
df <- complete %>% filter(mistake==0) %>% 
      mutate(L = log(lower), 
             U = log(upper), 
             se =(U-L)/(2*1.96),
             b=(L+U)/2,
             z=b/se) %>% 
      filter(!is.na(z), 
             ci.level == 0.95,
             mistake==0,
             source == "Abstract")
df_group <- group_by(df, pubmed)
```

```{r}
set.seed(2025)
df_keep_first <- group_by(df, pubmed) %>% filter(row_number() == 1) %>% ungroup()
df_keep_rand <- df %>% group_by(pubmed) %>%slice_sample(n = 1) %>%ungroup()
```

```{r}
table(df_keep_first$Year)
```

```{r}
years <- seq(2000, to=2018,by=1)
df_2000_2018 <- df_keep_first %>% filter(Year %in% years)
df_2000_2018 <- df_keep_rand %>% filter(Year %in% years)
````
```{r}
write_csv(df_2000_2018, "medline_2000_2018_random.csv")
```

```{r}
df_2000_2018_plos_one <- df_2000_2018 %>% filter(journal=="PLoS One")
```

```{r}
df_2018 <-  df_keep_rand %>% filter(Year ==2018)
```{r}
write_csv(df_2018, "medline_2018_random.csv")
```
year_cnt = count(df, Year)

L=log(complete$lower)

U=log(complete$upper)

se=(U-L)/(2*1.96)

b=(L+U)/2

z=b/se

ind=abs(z)<10

z = z[!is.na(z)]

n=sum(ind & !is.na(z))

hist(b[ind]/se[ind],breaks=100,xlab="z-value",ylab='', main=paste(n," z-values from Medline"))