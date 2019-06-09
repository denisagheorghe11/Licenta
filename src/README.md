## Description

 The python script data_extract.py its extracting the data from the MySQLdb as
python variables.
 The python script data_extract_to_xcels.py its extracting the table of the MySQLdb to an excel file for further processing in development mode.
 The python script data_prediction.py its procesing the data and determine the predictive models.

## Testing environment and scenario

There are two testing scenario's.

## First Scenario

TBD

## Second Scenario

This scenario collects the data from two phones into a MySQLdb of commmon radio parameters. one of the phone (UE1) is left into the best radio channel conditions of the LTE network while in the same time the second one (UE2) is moving between several rooms.


## Conclusion

TBD



## Sets of results:

                                      VAR_NAME    MIN_VALUE    MAX_VALUE  ...  DIST_NON_EVENT       WOE        IV
0                                 LCheadertype         12.0         12.0  ...           1.000  0.000000  0.000000
1                              LCheaderversion          0.0          0.0  ...           1.000  0.000000  0.000000
2                                  LCheaderxid          0.0          0.0  ...           1.000  0.000000  0.000000
3              LClcUeConfig0lcConfig0direction          2.0          2.0  ...           1.000  0.000000  0.000000
4                    LClcUeConfig0lcConfig0lcg          0.0          0.0  ...           1.000  0.000000  0.000000
5                   LClcUeConfig0lcConfig0lcid          1.0          1.0  ...           1.000  0.000000  0.000000
6                    LClcUeConfig0lcConfig0qci          1.0          1.0  ...           1.000  0.000000  0.000000
7          LClcUeConfig0lcConfig0qosBearerType          0.0          0.0  ...           1.000  0.000000  0.000000
8              LClcUeConfig0lcConfig1direction          2.0          2.0  ...           1.000  0.000000  0.000000
9                    LClcUeConfig0lcConfig1lcg          0.0          0.0  ...           1.000  0.000000  0.000000
10                  LClcUeConfig0lcConfig1lcid          2.0          2.0  ...           1.000  0.000000  0.000000
11                   LClcUeConfig0lcConfig1qci          1.0          1.0  ...           1.000  0.000000  0.000000
12         LClcUeConfig0lcConfig1qosBearerType          0.0          0.0  ...           1.000  0.000000  0.000000
13             LClcUeConfig0lcConfig2direction          1.0          1.0  ...           1.000  0.000000  0.000000
14                   LClcUeConfig0lcConfig2lcg          1.0          1.0  ...           1.000  0.000000  0.000000
15                  LClcUeConfig0lcConfig2lcid          3.0          3.0  ...           1.000  0.000000  0.000000
16                   LClcUeConfig0lcConfig2qci          1.0          1.0  ...           1.000  0.000000  0.000000
17         LClcUeConfig0lcConfig2qosBearerType          0.0          0.0  ...           1.000  0.000000  0.000000
18                           LClcUeConfig0rnti       5268.0       5268.0  ...           1.000  0.000000  0.000000
19          UEueConfig0ackNackRepetitionFactor          0.0          0.0  ...           1.000  0.000000  0.000000
20         UEueConfig0ackNackSimultaneousTrans          0.0          0.0  ...           1.000  0.000000  0.000000
21              UEueConfig0aperiodicCqiRepMode          3.0          3.0  ...           1.000  0.000000  0.000000
22               UEueConfig0betaOffsetACKIndex          0.0          0.0  ...           1.000  0.000000  0.000000
23               UEueConfig0betaOffsetCQIIndex          8.0          8.0  ...           0.875 -0.559616  0.729716
24               UEueConfig0betaOffsetCQIIndex          9.0          9.0  ...           0.125  1.386294  0.729716
25                UEueConfig0betaOffsetRIIndex          0.0          0.0  ...           1.000  0.000000  0.000000
26           UEueConfig0capabilitieshalfDuplex          0.0          0.0  ...           1.000  0.000000  0.000000
27       UEueConfig0capabilitiesintraSFHopping          1.0          1.0  ...           1.000  0.000000  0.000000
28        UEueConfig0capabilitiesresAllocType1          1.0          1.0  ...           1.000  0.000000  0.000000
29             UEueConfig0capabilitiestype2Sb1          1.0          1.0  ...           1.000  0.000000  0.000000
..                                         ...          ...          ...  ...             ...       ...       ...
121     eNBcellConfig0sliceConfigul0accounting          0.0          0.0  ...           1.000  0.000000  0.000000
122        eNBcellConfig0sliceConfigul0firstRb          0.0          0.0  ...           1.000  0.000000  0.000000
123             eNBcellConfig0sliceConfigul0id          0.0          0.0  ...           1.000 -0.143101  0.019080
124             eNBcellConfig0sliceConfigul0id          1.0          1.0  ...           0.000  0.000000  0.019080
125      eNBcellConfig0sliceConfigul0isolation          0.0          0.0  ...           1.000 -0.143101  0.019080
126      eNBcellConfig0sliceConfigul0isolation          1.0          1.0  ...           0.000  0.000000  0.019080
127          eNBcellConfig0sliceConfigul0label          0.0          0.0  ...           1.000  0.000000  0.000000
128         eNBcellConfig0sliceConfigul0maxmcs         20.0         20.0  ...           1.000  0.000000  0.000000
129     eNBcellConfig0sliceConfigul0percentage        100.0        100.0  ...           1.000  0.000000  0.000000
130       eNBcellConfig0sliceConfigul0priority          0.0          0.0  ...           1.000  0.000000  0.000000
131  eNBcellConfig0sliceConfigul0schedulerName          0.0          0.0  ...           1.000  0.000000  0.000000
132      eNBcellConfig0specialSubframePatterns          0.0          0.0  ...           1.000  0.000000  0.000000
133                  eNBcellConfig0srsBwConfig          0.0          0.0  ...           1.000  0.000000  0.000000
134                  eNBcellConfig0srsMacUpPts          0.0          0.0  ...           1.000  0.000000  0.000000
135            eNBcellConfig0srsSubframeConfig          0.0          0.0  ...           1.000  0.000000  0.000000
136           eNBcellConfig0subframeAssignment          0.0          0.0  ...           1.000  0.000000  0.000000
137                  eNBcellConfig0ulBandwidth          NaN          NaN  ...           0.000       NaN  0.729716
138                  eNBcellConfig0ulBandwidth         25.0         25.0  ...           0.875 -0.559616  0.729716
139                  eNBcellConfig0ulBandwidth         50.0        100.0  ...           0.125  1.386294  0.729716
140         eNBcellConfig0ulCyclicPrefixLength          0.0          0.0  ...           1.000  0.000000  0.000000
141                       eNBcellConfig0ulFreq       2565.0       2565.0  ...           1.000  0.000000  0.000000
142                 eNBcellConfig0ulPuschPower        -96.0        -50.0  ...           1.000  0.000000  0.000000
143                 eNBcellConfig0ulPuschPower          NaN          NaN  ...           0.000       NaN  0.000000
144                 eNBcellConfig0ulPuschPower          NaN          NaN  ...           0.000       NaN  0.000000
145                                   eNBeNBId  234881024.0  234881024.0  ...           1.000 -0.143101  0.019080
146                                   eNBeNBId  234881025.0  234881025.0  ...           0.000  0.000000  0.019080
147                              eNBheadertype          8.0          8.0  ...           1.000 -0.143101  0.019080
148                              eNBheadertype          9.0          9.0  ...           0.000  0.000000  0.019080
149                           eNBheaderversion          0.0          0.0  ...           1.000  0.000000  0.000000
150                               eNBheaderxid          0.0          0.0  ...           1.000  0.000000  0.000000


Figure 1. 

![Imgur](https://i.imgur.com/rXniAk6.png)
