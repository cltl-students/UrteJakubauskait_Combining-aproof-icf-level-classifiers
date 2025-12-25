"""
This script filters a training dataset to include only sentences that contain 
specific linguistic words or expressions (such as intensifiers, modifiers, or 
other target elements). The filtered dataset is then saved as both a pickle 
and a CSV file.

Firstly, it loads the training dataset from a pickle file (for example,
'train.pkl'). Then, it defines a list of target words or expressions to filter
for. Then, it creates a regex pattern that matches any of the target words as
whole words and filters the dataset to keep only sentences in the 'text' column
that contain at least one target word. Finally, it saves the filtered dataset
as a new pickle file (for example, "filtered_train_ADM.pkl") and CSV file
(for example, "filtered_dataset_ADM.csv").
"""

import pandas as pd 
import re

words = ['aanzienlijk', 'aardig', 'aardige', 'afgebakend', 'afgebakende', 'alledaags', 
         'amper', 'apart', 'aparte', 'beetje', 'begrensd', 'begrensde', 'behoorlijk', 
         'behoorlijk wat', 'behoorlijke', 'beperkt', 'beperkte', 'beroerd', 'beroerde', 
         'bijna', 'bijna niet', 'bijzonder', 'bijzondere', 'billijk', 'billijke', 'bruikbaar', 
         'compleet', 'degelijk', 'degelijke', 'doorsnee', 'een beetje', 'een boel', 
         'een fractie', 'een hoop', 'eindig', 'eindige', 'enigszins', 'enorm', 'enorme', 
         'erg', 'erge', 'ernstig', 'ernstige', 'fantastisch', 'fantastische', 'feitelijk', 
         'flink', 'flink wat', 'flinke', 'gangbaar', 'gangbare', 'gebrekkig', 'gebrekkige', 
         'gebruikelijk', 'geen', 'geen enkel ding', 'geeneen', 'geenszins', 'geheel', 'gelimiteerd', 
         'gelimiteerde', 'gemiddeld', 'gemiddelde', 'gering', 'geringe', 'geringste', 'geweldig', 
         'geweldige', 'gewone', 'gigantisch', 'gigantische', 'goed', 'goede', 'grootste', 'haast', 
         'heel', 'heel wat', 'heftig', 'heftige', 'helemaal', 'helemaal niets', 'hinderlijk', 
         'hinderlijke', 'hoogste', 'hoogstens', 'iets', 'ietsje', 'ietwat', 'in geringe mate', 
         'in grote mate', 'in hoge mate', 'in kleine mate', 'in zekere mate', 'ingewikkeld', 
         'ingewikkelde', 'ingrijpend', 'ingrijpende', 'kleinste', 'kolossaal', 'kolossale', 
         'krachtig', 'krachtige', 'laagste', 'lastig', 'lastige', 'lichtjes', 'logisch', 
         'logische', 'maximaal', 'maximale', 'menig', 'menige', 'met moeite', 'minimaal', 
         'minimale', 'minstens', 'modaal', 'moeilijk', 'moeilijke', 'nagenoeg', 'nauwelijks', 
         'niet', 'niet één', 'niets', 'niks', 'nimmer', 'nooit', 'normaal', 'normale', 'nul', 
         'onaanzienlijk', 'onaanzienlijke', 'onalledaags', 'onbeduidend', 'onbeduidende', 'onbegrensd', 
         'onbegrensde', 'onbelangrijk', 'onbelangrijke', 'onbeperkt', 'onbeperkte', 'onbillijk', 
         'onbillijke', 'oncompleet', 'oneindig', 'oneindige', 'ongebruikelijk', 'ongeheel', 
         'ongelimiteerd', 'ongelimiteerde', 'ongewone', 'onhelemaal', 'onklein', 'onkleine', 
         'onlogisch', 'onlogische', 'onrechtvaardig', 'onrechtvaardige', 'onredelijk', 'onredelijke', 
         'onstandaard', 'onterecht', 'onterechte', 'ontotaal', 'ontzettend', 'ontzettende', 'onvolkomen', 
         'onvolledig', 'op z’n meest', 'op z’n minst', 'opmerkelijk', 'opmerkelijke', 'positief', 
         'positieve', 'prachtig', 'prachtige', 'praktisch', 'prima', 'rechtvaardig', 'rechtvaardige', 
         'redelijk', 'redelijke', 'reusachtig', 'reusachtige', 'schitterend', 'schitterende', 'slecht', 
         'slechte', 'snufje', 'speciaal', 'standaard', 'sterk', 'stevig', 'stevige', 'subliem', 'super', 
         'talloze', 'talrijk', 'talrijke', 'ten hoogste', 'ten minste', 'terecht', 'terechte', 
         'ternauwernood', 'tikje', 'tikkeltje', 'totaal', 'uiterste', 'uitstekend', 'uitstekende', 
         'uitzonderend', 'uitzonderlijk', 'uitzonderlijke', 'uniek', 'veel', 'vele', 'vervelend', 
         'vervelende', 'volkomen', 'volledig', 'voortreffelijk', 'voortreffelijke', 'vreselijk', 
         'vreselijke', 'vrijwel', 'waardeloos', 'waardeloze', 'wat', 'zeer', 'zelden', 'zo goed als', 
         'zorgwekkend', 'zorgwekkende', 'zwaar', 'zwak', 'zwakke']

df = pd.read_pickle('train.pkl')

pattern = r'\b(?:' + '|'.join(re.escape(word) for word in words) + r')\b'

filtered_df = df[df['text'].str.contains(pattern, case=False, regex=True)]

filtered_df.to_pickle('filtered_train_ADM.pkl')
filtered_df.to_csv('filtered_dataset_ADM.csv', index=False)
