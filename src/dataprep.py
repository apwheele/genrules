'''
This script is to help
with basic data preparation
with the nuMoM2b dataset
'''

import pandas as pd
import numpy as np

# location of the data in this repository (not saved to Github!)
data_loc = './data/nuMoM2b_Dataset_NICHD Data Challenge.csv'

# This does dummy variables for multiple columns
# Here used for drug codes, but could be used for ICD codes
# In health claims data (order does not matter)
def encode_mult(df, cols, encode=None, base="V", cum=False, missing_drop=True):
    res_counts = []
    if not encode:
        tot_encode = pd.unique(df[cols].values.ravel())
        if missing_drop:
            check = pd.isnull(tot_encode)
            tot_encode = tot_encode[~check]
    else:
        tot_encode = encode
    var_names = []
    for i,v in enumerate(tot_encode):
        var_name = base + str(i)
        res_count = (df[cols] == v).sum(axis=1)
        res_counts.append( res_count )
        var_names.append(var_name)
    res_count_df = pd.concat(res_counts, axis=1)
    res_count_df.columns = var_names
    if cum:
        return res_count_df, list(tot_encode)
    else:
        return 1*(res_count_df > 0), list(tot_encode)

# This function takes the recode dict and columns
def recode_vars(codes, columns, data):
    data[columns] = data[columns].replace(codes)

#######################################################
# These are the different variable sets

'''
A09A03a - pre-term birth [784 yes]
A09A03b - maternal hypertensive disorder [1464 yes]
CBAC01 - neonatal morbidities (general) [1872 yes]

1 = yes
2 = no
others = missing data

pOUTCOME

1	Live birth
2	Stillbirth
3	Fetal demise < 20 weeks
4	Elective termination
5	Indicated termination
D	Don't know
R	Refused

CMAJ01d1 - readmission infection 14 days
CMAJ01d2 - readmission preclampsia
CMAJ01d3 - readmission bleeding
CMAJ01d4 - readmission depression

CMAE04a1c - postpartum depression

CBAB01 - Documented infection

A09A03b3 - maternal hypertensive disorder - Preeclampsia/HELLP/eclampsia

'''

outcomes = ['A09A03a','A09A03b','CBAC01','pOUTCOME',
            'CMAJ01d1','CMAJ01d2','CMAJ01d3','CMAJ01d4',
            'CMAE04a1c','CBAB01','A09A03b3']

'''
CMAE04 - mental health condition
CMAE04a1a - depression prior
CMAE04a2a - anxiety prior
CMAE04a3a - bipolar prior
CMAE04a4a - PTSD prior
CMAE04a5a - Schizophrenia prior
CMAE04a6a - treated for other mental health

0/1 (checked/not checked)
'''

mental_health = ['CMAE04','CMAE04a1a','CMAE04a2a','CMAE04a3a',
                 'CMAE04a4a','CMAE04a5a','CMAE04a6a']

'''
CMAE03 - diabetes ever diagnosed (1 yes before, 2 yes during preg, 3 no)

CMDA01 - hypertension prior 20 weeks
CMDA02 - proteinuria (protein urine) prior 20 weeks

[Yes = 1, No = 2]

V1AD06	(V1A) Have you had any 'flu-like illnesses,' 'really bad colds,' fever, a rash, or any muscle or joint aches since you became pregnant?

V1AD12a	(V1A) Previous surgeries - Cervical surgery - Cone
V1AD12b	(V1A) Previous surgeries - Cervical surgery - LEEP
V1AD12c	(V1A) Previous surgeries - Cervical surgery - Cryotherapy
V1AD12d	(V1A) Previous surgeries - Myomectomy
V1AD12e	(V1A) Previous surgeries - Abdominal surgery excluding uterine surgery
V1AD13	(V1A) Have you ever had a blood transfusion?

[Yes = 1, No = 2]

'''

health_cond = ['CMAE03','CMDA01','CMDA02','V1AD06','V1AD12a','V1AD12b',
               'V1AD12c','V1AD12d','V1AD12e','V1AD13']

'''
V1AD14	(V1A) Do you use a continuous positive airway pressure (CPAP) or other breathing machine when you sleep?
V1AD15	(V1A) Do you have asthma?
V1AD15a	(V1A) Are you currently using oral steroid tablets or liquids (such as prednisone, deltasone, orasone, prednisolone, prelone, medrol, or solumedrol) for treatment of asthma?
V1AD15a1	(V1A) Asthma treatment - Did your doctor ask you to take this every day?
V1AD15a2	(V1A) Asthma treatment - Did you start taking the steroid tablets or liquid more than 14 days ago?
V1AD16	(V1A) Do you receive oxygen therapy, either during the day or at night, for any medical condition(s)?

[Yes = 1, No = 2]
'''

asthma = ['V1AD14','V1AD15','V1AD15a','V1AD15a1','V1AD15a2','V1AD16']

'''
V1AD02a	(V1A) Before you got pregnant, did a doctor, nurse, or other health care worker talk to you about the following regarding how to prepare for a healthy pregnancy and baby? - Taking vitamins with folic acid before pregnancy
V1AD02b	(V1A) Before you got pregnant, did a doctor, nurse, or other health care worker talk to you about the following regarding how to prepare for a healthy pregnancy and baby? - Being a healthy weight before pregnancy
V1AD02c	(V1A) Before you got pregnant, did a doctor, nurse, or other health care worker talk to you about the following regarding how to prepare for a healthy pregnancy and baby? - Getting your vaccines updated before pregnancy
V1AD02d	(V1A) Before you got pregnant, did a doctor, nurse, or other health care worker talk to you about the following regarding how to prepare for a healthy pregnancy and baby? - Visiting a dentist or dental hygienist before pregnancy
V1AD02e	(V1A) Before you got pregnant, did a doctor, nurse, or other health care worker talk to you about the following regarding how to prepare for a healthy pregnancy and baby? - Getting counseling for any genetic diseases that run in your family
V1AD02f	(V1A) Before you got pregnant, did a doctor, nurse, or other health care worker talk to you about the following regarding how to prepare for a healthy pregnancy and baby? - Controlling any medical conditions, such as diabetes or high blood pressure
V1AD02g	(V1A) Before you got pregnant, did a doctor, nurse, or other health care worker talk to you about the following regarding how to prepare for a healthy pregnancy and baby? - Getting counseling or treatment for depression or anxiety
V1AD02h	(V1A) Before you got pregnant, did a doctor, nurse, or other health care worker talk to you about the following regarding how to prepare for a healthy pregnancy and baby? - The safety of using prescription or over-the-counter medicines during pregnancy
V1AD02i	(V1A) Before you got pregnant, did a doctor, nurse, or other health care worker talk to you about the following regarding how to prepare for a healthy pregnancy and baby? - How smoking during pregnancy can affect a baby
V1AD02j	(V1A) Before you got pregnant, did a doctor, nurse, or other health care worker talk to you about the following regarding how to prepare for a healthy pregnancy and baby? - How drinking alcohol during pregnancy can affect a baby
V1AD02k	(V1A) Before you got pregnant, did a doctor, nurse, or other health care worker talk to you about the following regarding how to prepare for a healthy pregnancy and baby? - How using illegal drugs during pregnancy can affect a baby
V1AD03	(V1A) Was this pregnancy planned?
V1AD05	(V1A) Is this pregnancy desired?

[Yes = 1, No = 2]
'''

health_discussion = ['V1AD02a','V1AD02b','V1AD02c','V1AD02d','V1AD02e',
                     'V1AD02f','V1AD02g','V1AD02h','V1AD02i','V1AD02j',
                     'V1AD02k','V1AD03','V1AD05']

'''
urgalertyn	C1. Did review result in urgent alerts? (No/Yes)
alertahi50	C1b1. Apnea-hypopnea index (AHI) > 50 (No/Yes)
alerthypoyn	B2. Severe hypoxemia (No/Yes)
alerthyporest	B2a. Baseline O2 sat < 88% (checkbox)
alerthyposleep	B2b. O2 sat during sleep <90% for >10% of sleep time (checkbox)
alertecgyn	B3. Specific heart rate and/or ECG finding (No/Yes)
alertecghr40	B3a. HR for >2 continuous minutes is <40 bpm (checkbox)
alertecghr150	B3b. HR for >2 continuous minutes is >150 bpm (checkbox)

0 = No, 1 = Yes
'''

sleep_alert = ['urgalertyn','alertahi50','alerthypoyn','alerthyporest',
               'alerthyposleep','alertecgyn','alertecghr40','alertecghr150']

'''
V2AE06a	(V2A) Have any of your biological mother, sisters, half-sisters, or female first cousins ever had pregnancy complications - Delivery of a child more than 3 weeks before the expected due date
V2AE06b	(V2A) Have any of your biological mother, sisters, half-sisters, or female first cousins ever had pregnancy complications - Delivery of a child weighing less than 5 lb 8 oz (or 2500 grams)
V2AE06c	(V2A) Have any of your biological mother, sisters, half-sisters, or female first cousins ever had pregnancy complications - Spontaneous preterm delivery (<37 weeks)
V2AE06d	(V2A) Have any of your biological mother, sisters, half-sisters, or female first cousins ever had pregnancy complications - Early or preterm rupture of the membranes
V2AE06e	(V2A) Have any of your biological mother, sisters, half-sisters, or female first cousins ever had pregnancy complications - Preeclampsia, eclampsia, toxemia or pregnancy induced hypertension
V2AE06f	(V2A) Have any of your biological mother, sisters, half-sisters, or female first cousins ever had pregnancy complications - Induction of labor due to low amniotic fluid or poor fetal growth
V2AE06g	(V2A) Have any of your biological mother, sisters, half-sisters, or female first cousins ever had pregnancy complications - Stillbirth
V2AE06h	(V2A) Have any of your biological mother, sisters, half-sisters, or female first cousins ever had pregnancy complications - Delivery of an infant with a birth defect
V2AE06i	(V2A) Have any of your biological mother, sisters, half-sisters, or female first cousins ever had pregnancy complications - Other pregnancy complication

1 = Yes, 2 = No
'''

fam_complications = ['V2AE06a','V2AE06b','V2AE06c','V2AE06d','V2AE06e',
                     'V2AE06f','V2AE06g','V2AE06h','V2AE06i']

'''
Social Support

V1GA01	(V1G) Social Support - There is a special person who is around when I am in need
V1GA02	(V1G) Social Support - There is a special person with whom I can share my joys and sorrows
V1GA03	(V1G) Social Support - My family really tries to help me
V1GA04	(V1G) Social Support - I get the emotional help and support I need from my family
V1GA05	(V1G) Social Support - I have a special person who is a real source of comfort to me
V1GA06	(V1G) Social Support - My friends really try to help me
V1GA07	(V1G) Social Support - I can count on my friends when things go wrong
V1GA08	(V1G) Social Support - I can talk about my problems with my family
V1GA09	(V1G) Social Support - I have friends with whom I can share my joys and sorrows
V1GA10	(V1G) Social Support - There is a special person in my life who cares about my feelings
V1GA11	(V1G) Social Support - My family is willing to help me make decisions
V1GA12	(V1G) Social Support - I can talk about my problems with my friends

1	Very strongly disagree
2	Strongly disagree
3	Mildly disagree
4	Neutral
5	Mildly agree
6	Strongly agree
7	Very strongly agree
'''

social_support = ['V1GA01','V1GA02','V1GA03','V1GA04','V1GA05','V1GA06',
                  'V1GA07','V1GA08','V1GA09','V1GA10','V1GA11','V1GA12']

'''
SmokeCat1	Ever used tobacco (V1AG04)
SmokeCat2	Smoked tobacco in the 3 months prior to pregnancy (V1AG05)
SmokeCat3	Among those who smoked in the prior 3 months, how many cigarettes per day (V1AG05a)
'''

smoking = ['SmokeCat1','SmokeCat2','SmokeCat3']

'''
Alcohol/Drugs

V1AG12a	(V1A) Every used any of these drugs - Marijuana (THC)
V1AG12b	(V1A) Every used any of these drugs - Cocaine
V1AG12c	(V1A) Every used any of these drugs - Prescription narcotics that were not prescribed for you
V1AG12d	(V1A) Every used any of these drugs - Heroin

V1AG12e	(V1A) Every used any of these drugs - Methadone
V1AG12f	(V1A) Every used any of these drugs - Amphetamines (speed) not prescribed for you
V1AG12g	(V1A) Every used any of these drugs - Inhalants not prescribed for you
V1AG12h	(V1A) Every used any of these drugs - Hallucinogens
V1AG12i	(V1A) Every used any of these drugs - Other
V1AG14	(V1A) Have you ever considered yourself to be addicted to these drugs?

[1 = Yes, 2 = No]
'''

substances = ['V1AG12a','V1AG12b','V1AG12c',
              'V1AG12d','V1AG12e','V1AG12f','V1AG12g','V1AG12h','V1AG12i',
              'V1AG14']

'''
AgeCat_V1	Age category (years) at visit 1 calculated from DOB (V1AB01) and the form completion date

1	13-17
2	18-34
3	35-39
4	>= 40


CRace	Race/ethnicity combined categories derived from V1AF05 and V1AF07a-g

1	Non-Hispanic White
2	Non-Hispanic Black
3	Hispanic
4	Asian
5	Other

BMI_Cat	BMI category (kg/m^2) at visit 1 calculated from height and weight (form V1B)

1	< 18.5 (underweight)
2	18.5 - < 25 (normal weight)
3	25 - < 30 (overweight)
4	30 - < 35 (obese)
5	>= 35 (morbidly obese)

Education	Education status attained (V1AF02)

1	Less than HS grad
2	HS grad or GED
3	Some college
4	Assoc/Tech degree
5	Completed college
6	Degree work beyond college

poverty	Poverty category based on income (V1AF14) and household size (V1AF13) relative to 2013 federal poverty guidelines

1	> 200% of fed poverty level
2	100-200% of fed poverty level
3	< 100% of fed poverty level
'''

demo = ['AgeCat_V1','CRace','BMI_Cat','Education','poverty']

'''
Ins_Govt	Health care paid for by govt insurance (V1AF15a)
Ins_Mil	Health care paid for by military insurance (V1AF15b)
Ins_Comm	Health care paid for by commercial health insurance (V1AF15c)
Ins_Pers	Health care paid for by personal income (V1AF15d)
Ins_Othr	Health care paid for by other (V1AF15e)

1 = Yes, 2 = No
'''

insurance = ['Ins_Govt','Ins_Mil','Ins_Comm','Ins_Pers','Ins_Othr']

'''
V1AF03	(V1A) Do you currently have a partner or 'significant other'?
V1AF03a1	(V1A) What support do you expect your partner to give you during this pregnancy? - Emotional support
V1AF03a2	(V1A) What support do you expect your partner to give you during this pregnancy? - Financial support
V1AF03a3	(V1A) What support do you expect your partner to give you during this pregnancy? - To be present for my prenatal visits
V1AF03a4	(V1A) What support do you expect your partner to give you during this pregnancy? - To be present for the delivery
V1AF03b	(V1A) Are you currently living with your partner?
V1AF03c	(V1A) Is your current partner the biological father of the baby?

1	Yes
2	No
3	Not done/none recorded
D	Don't know
M	Missing
N	Not applicable
R	Refused
'''

partner = ['V1AF03','V1AF03a1','V1AF03a2','V1AF03a3','V1AF03a4','V1AF03b','V1AF03c']

"""
Ultrasound abnormalities [Normal/Abnormal/Not Reported]

'CUAB01a',	(CUA) Brain region - Calvarium
'CUAB01b',	(CUA) Brain region - Falx
'CUAB01c',	(CUA) Brain region - Cavum septi pellucidi
'CUAB01d',	(CUA) Brain region - Lateral ventricles
'CUAB01e',	(CUA) Brain region - Choroid plexus
'CUAB01f',	(CUA) Brain region - Thalami
'CUAB01g',	(CUA) Brain region - Cerebellum
'CUAB01h',	(CUA) Brain region - Posterior fossa
'CUAB01i',	(CUA) Brain region - Cisterna magna
'       ',
'CUAC01a',	(CUA) Head/neck region - Lenses
'CUAC01b',	(CUA) Head/neck region - Orbits
'CUAC01c',	(CUA) Head/neck region - Profile
'CUAC01d',	(CUA) Head/neck region - Nasal bone
'CUAC01e',	(CUA) Head/neck region - Upper lip
'       ',
'CUAD01a',	(CUA) Chest region - Cardiac axis
'CUAD01b',	(CUA) Chest region - Four chamber view
'CUAD01c',	(CUA) Chest region - RVOT
'CUAD01d',	(CUA) Chest region - LVOT
'CUAD01e',	(CUA) Chest region - Three vessel view
'CUAD01f',	(CUA) Chest region - Lungs
'CUAD01g',	(CUA) Chest region - Diaphragm
'       ',
'CUAE01a',	(CUA) Abdomen region - Situs
'CUAE01b',	(CUA) Abdomen region - Ventral wall / cord
'CUAE01c',	(CUA) Abdomen region - Stomach
'CUAE01d',	(CUA) Abdomen region - Kidneys
'CUAE01e',	(CUA) Abdomen region - Gallbladder
'CUAE01f',	(CUA) Abdomen region - Bladder
'CUAE01g',	(CUA) Abdomen region - Umbilical arteries
'CUAE01h',	(CUA) Abdomen region - Bowel
'CUAE01i',	(CUA) Abdomen region - Genitalia
'       ',
'CUAF01a',	(CUA) Extremities region - Upper extremities
'CUAF01b',	(CUA) Extremities region - Lower extremities
'       ',
'CUAG01a',	(CUA) Spine region - Cervical
'CUAG01b',	(CUA) Spine region - Thoracic
'CUAG01c',	(CUA) Spine region - Lumbar
'CUAG01d',	(CUA) Spine region - Sacral

CUBB01	(CUB) Were any structural abnormalities detected? [Yes/no]

1	Normal
2	Abnormal
3	Not reported/Unknown
"""

ultrasound_abnormal = ['CUBB01',
'CUAB01a',
'CUAB01b',
'CUAB01c',
'CUAB01d',
'CUAB01e',
'CUAB01f',
'CUAB01g',
'CUAB01h',
'CUAB01i',
'CUAC01a',
'CUAC01b',
'CUAC01c',
'CUAC01d',
'CUAC01e',
'CUAD01a',
'CUAD01b',
'CUAD01c',
'CUAD01d',
'CUAD01e',
'CUAD01f',
'CUAD01g',
'CUAE01a',
'CUAE01b',
'CUAE01c',
'CUAE01d',
'CUAE01e',
'CUAE01f',
'CUAE01g',
'CUAE01h',
'CUAE01i',
'CUAF01a',
'CUAF01b',
'CUAG01a',
'CUAG01b',
'CUAG01c',
'CUAG01d']

'''
CMEA01B1A_INT - UTI [negative before delivery]
CMEA02A1A_INT - Sepsis blood culture
CMEA02B1A_INT - Sepsis gram strain
CMEA02C1A_INT - Sepsis spinal fluid
CMEA03A1_INT - Pneumonia 
CMEA04A2_INT - other maternal infections

INT is days, so negative values are before birth
'''

#Too few of these to worry about, only a handful
infections = ['CMEA01B1A_INT','CMEA02A1A_INT','CMEA02B1A_INT',
              'CMEA02C1A_INT','CMEA03A1_INT','CMEA04A2_INT']

"""
CMDA03a_Check - done/not done
CMDA03a1 - serum creatinine [For adult women, 0.59 to 1.04 mg/dL (52.2 to 91.9 micromoles/L), https://www.mayoclinic.org/tests-procedures/creatinine-test/about/pac-20384646]

CMDA03b_Check
CMDA03b1 - AST Liver check [Females: 9 to 32 units/L, https://www.webmd.com/a-to-z-guides/aspartate_aminotransferse-test]

CMDA03c_Check
CMDA03c1 - uric acid [For females, it’s over 6 mg/dL is high, https://www.webmd.com/arthritis/uric-acid-blood-test]

CMDA09g_Check
CMDA09g1 - lowest serum platelet count [A normal platelet count ranges from 150,000 to 450,000 platelets per microliter of blood, https://www.hopkinsmedicine.org/health/conditions-and-diseases/what-are-platelets-and-why-are-they-important]

CMDA09i_Check
CMDA09i1 -  highest serum lactate dehydrogenase [usually range between 140 units per liter (U/L) to 280 U/L for adults, https://www.webmd.com/a-to-z-guides/lactic-acid-dehydrogenase-test]

CMDA09j_Check
CMDA09j1 -  highest serum total bilirubin [Normal results for a total bilirubin test are 1.2 milligrams per deciliter (mg/dL) for adults, low are ok, high is bad, https://www.mayoclinic.org/tests-procedures/bilirubin/about/pac-20393041]
"""

check_vars = ['CMDA03a_Check','CMDA03b_Check','CMDA03c_Check','CMDA09g_Check',
              'CMDA09i_Check','CMDA09j_Check']
lab_vars = [i.split("_")[0]+"1" for i in check_vars]

low_high = {'CMDA03a1':(0.59,1.04),
            'CMDA03b1':(9,32),
            'CMDA03c1':(0,6),
            'CMDA09g1':(150,450),
            'CMDA09i1':(140,280),
            'CMDA09j1':(0,1.2)}


# Need to make these into dummy variables
# If before/during
'''
Drug_Timing	1	> 2 Months before pregnancy
Drug_Timing	2	During 2 months before pregnancy
Drug_Timing	3	During Pregnancy, before first visit
Drug_Timing	4	Before second visit
Drug_Timing	5	Before third visit
Drug_Timing	6	Before delivery
Drug_Timing	D	Don't know

Drug Codes
101	Narcotic 
102	NSAID 
103	Triptans 
109	Other analgesics 
111	Nitrofurantoin 
112	Penicillins 
113	Metronidazole 
114	Macrolides 
115	Sulfa drugs 
116	Aminoglycosides 
117	Clindamycin 
118	Cephalosporins 
119	Other antibiotic 
121	Systemic antifungals 
122	Vaginal antifungals 
131	Antivirals - flu 
132	Antivirals - herpes 
139	Antivirals - other 
141	Warfarin 
142	Unfractionated heparin 
143	LMWH 
149	Other anticoagulant 
150	Antipsychotics 
161	MAOI 
162	TCA 
163	SSRI 
164	SNRI 
165	NDRI 
166	Augmenter drug 
169	Other antidepressant 
171	GABA analogs 
172	Benzodiazepines 
173	Caboxamides (carbamazepine) 
174	Fructose derivative (topiramate) 
175	Hydantoins 
176	Triazines (lamotrigine) 
177	Valproate 
179	Other anticonvulsant 
181	Nifedipine/Procardia 
182	Indomethacin/Indocin 
183	Magnesium Sulfate 
184	Terbutaline/Brethine 
189	Other tocolytic 
191	Methyldopa (Aldomet) 
192	Labetolol 
193	Ca-Channel Blocker 
194	Beta-Blocker 
195	ACE-Inhibitor 
199	Other antihypertensives 
200	Diuretics 
211	Motility agents 
212	Anti-nausea 
213	SHT3 antagonist 
214	Other NVP agents 
215	PPIs 
216	H2 receptor agonists 
219	Other GI agents 
220	Chemotherapeutics 
230	Steroids (systemic) 
240	Hormonal contraceptives 
250	Progesterone (for purpose other than contraception) 
261	Antithyroids (overactive) 
262	Thyroid replacement (under-active) 
271	Bronchodilator 
272	Inhaled steroid 
273	Immune modulator 
280	Decongestants 
290	Antihistamines 
300	Combined antihistamines / decongestant 
310	Combined antihistamines / decongestant / analgesic 
320	Anti-anxiety medications 
330	Mood stabilizers (lithium) 
341	Insulin 
342	Metformin 
343	Glyburide 
349	Other anti-diabetic medication 
499	Other medication 
510	Prenatal Multivitamin 
520	Other Multivitamin 
530	Additional Iron 
540	Additional Folate 
599	Other Vitamin 
610	Influenza (seasonal/novel) 
620	Hepatitis B 
630	Rubella (German measles) 
640	MMR 
650	Varicella-zoster immune globulin (VZIG chicken pox) 
660	Pertussis 
699	Other vaccine 
-8	Don't know
'''

drug_codes = ['DrugCode'] + ['DrugCode_' + str(i+1) for i in range(27)]
drugtime_codes = ['VXXC01g'] + ['VXXC01g_' + str(i+1) for i in range(27)]

# These are the sets all put together
all_vars  = outcomes + mental_health + health_cond
all_vars += check_vars + lab_vars + insurance
all_vars += ultrasound_abnormal + demo + partner
all_vars += substances + social_support + fam_complications
all_vars += drug_codes + drugtime_codes + sleep_alert
all_vars += health_discussion + asthma + smoking

# Loads the full data frame of these variables
full_dat = pd.read_csv(data_loc, usecols=all_vars, dtype=str)

#############
# Drug Prep, 2 months before pregnancy

not2m = full_dat[drugtime_codes] != '2'
drug_dat = full_dat[drug_codes].copy()
drug_dat.iloc[not2m] = np.NaN

drug_dummy, drug_encode = encode_mult(full_dat, drug_codes, base="Drug", cum=False)
drug_dummyvars = ["Drug_" + i for i in drug_encode]
drug_dummy.columns = drug_dummyvars

full_dat[drug_dummyvars] = drug_dummy
all_vars += drug_dummyvars
#############


#############
# Infections at least 60 days before birth
# Too few to worry about
#full_dat[infections].describe()
#############

#############
# Insurance type
ins_var = pd.Series('Ins_No',full_dat.index)
for i in insurance:
   check = full_dat[i] == '1'
   ins_var[check] = i

full_dat['ins_type'] = ins_var
ins_type = ['ins_type']

#all_vars += ins_type
demo += ins_type # just add it into demo variables
#############

#############
# Outlier lab results -1 too low, 1 is too high

lab_outlier = []
lab_ovars = []
for lv in lab_vars:
    num = pd.to_numeric(full_dat[lv],errors='coerce')
    low = (num < low_high[lv][0])
    high = (num > low_high[lv][1])
    res = pd.Series(0,full_dat.index)
    res[low] = -1
    res[high] = 1
    lab_outlier.append(res)
    lab_ovars.append('Out_' + lv)

lab_odf = pd.concat(lab_outlier, axis=1)
lab_odf.columns = lab_ovars

full_dat[lab_ovars] = lab_odf
all_vars += lab_ovars
#############


# Function to return data without missing for given
# outcome variable

def prep_dat(out,rhs):
    # If variables in rhs are not in current full_dat, reread base
    # and include
    if out is None:
        outl = []
    else:
        outl = [out]
    ro = set(rhs + outl)
    fd = set(list(full_dat))
    diff = list(ro - fd)
    new_dat = full_dat.copy()
    if len(diff) > 0:
        upd = pd.read_csv(data_loc, usecols=diff, dtype=str)
        new_dat[diff] = upd
    # If outcomes are not in my list, present a warning message
    if out in outcomes:
        pass
    else:
        print('\nWarning!!!')
        print(f'The outcome variable {out} is not in my list of outcomes.')
        print('Please encode the outcome variable as integer 0/1 before')
        print('being passed to the genrules() class')
    if out is None:
        subset = new_dat.copy()
    elif out[0:3] == 'CMA':
        subset = new_dat[new_dat[out].isin(['0','1'])].copy()
    elif out == 'A09A03b3':
        subset = new_dat.copy()
    elif out in outcomes:
        subset = new_dat[new_dat[out].isin(['1','2'])].copy()
    else:
        subset = new_dat.copy()
    # If poutcome, stillbirth is 2, others 1 is bad
    if out is None:
        pass
    elif out == 'pOUTCOME':
        subset[out] = 1*(subset[out] == '2')
    else:
        subset[out] = 1*(subset[out] == '1')
    subset.reset_index(drop=True,inplace=True)
    # Only returning specified variables
    if out is None:
        return subset[rhs]
    else:
        return subset[[out] + rhs]




