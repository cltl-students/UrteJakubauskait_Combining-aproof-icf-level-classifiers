"""
This script deduplicates and sorts a list of Dutch linguistic intensity words 
and expressions, including modifiers of degree, quantity, and negation. 

It converts the list of words into a set to remove duplicates. Then, it converts
the set back to a sorted list in alphabetical order. Finally, it prints the
sorted, deduplicated list.
"""

words = [
    "amper", "nauwelijks", "ternauwernood", "zelden", "met moeite",
    "beetje", "wat", "iets", "tikje", "tikkeltje", "snufje",
    "beperkt", "beperkte", "onbeperkt", "onbeperkte", "begrensd", 
    "begrensde", "onbegrensd", "onbegrensde", "eindig", "eindige", 
    "oneindig", "oneindige", "gelimiteerd", "gelimiteerde", "ongelimiteerd", 
    "ongelimiteerde", "afgebakend", "afgebakende", "bijna", "nagenoeg", 
    "haast", "zo goed als", "vrijwel", "bijzonder", "bijzondere", 
    "speciaal", "uitzonderlijk", "uitzonderlijke", "opmerkelijk",
    "opmerkelijke", "apart", "aparte", "uniek", "uitzonderend", "enigszins", 
    "een beetje", "lichtjes", "ietwat", "in zekere mate", "enorm", 
    "enorme", "gigantisch", "gigantische", "reusachtig", "reusachtige", 
    "kolossaal", "kolossale", "ontzettend", "ontzettende", "erg", "erge", 
    "heel", "zeer", "vreselijk", "vreselijke", "flink", "flinke", "ernstig", 
    "ernstige", "zwaar", "zorgwekkend", "zorgwekkende", "heftig", "heftige", 
    "ingrijpend", "ingrijpende", "behoorlijk", "behoorlijke", "stevig", 
    "stevige", "aardig", "aardige", "krachtig", "krachtige", "gemiddeld", 
    "gemiddelde", "doorsnee", "modaal", "normaal", "normale", "gangbaar", 
    "gangbare", "gering", "geringe", "onbeduidend", "onbeduidende", 
    "onbelangrijk", "onbelangrijke", "onaanzienlijk", "onaanzienlijke", 
    "onklein", "onkleine", "geweldig", "geweldige", "fantastisch", 
    "fantastische", "prachtig", "prachtige", "schitterend", "schitterende", 
    "super", "gewone", "ongewone", "alledaags", "onalledaags", "gebruikelijk", 
    "ongebruikelijk", "standaard", "onstandaard", "goed", "goede", "prima", 
    "degelijk", "degelijke", "uitstekend", "uitstekende", "voortreffelijk", 
    "voortreffelijke", "positief", "positieve", "subliem", "heel wat", 
    "flink wat", "behoorlijk wat", "een boel", "veel", "in geringe mate", 
    "beperkt", "in kleine mate", "in grote mate", "sterk", "in hoge mate", 
    "aanzienlijk", "lastig", "lastige", "moeilijk", "moeilijke", "vervelend", 
    "vervelende", "ingewikkeld", "ingewikkelde", "hinderlijk", "hinderlijke",
    "maximaal", "hoogstens", "ten hoogste", "op z'n meest", "maximale", 
    "hoogste", "uiterste", "grootste", "minimaal", "ten minste", "op z'n minst", 
    "minstens", "minimale", "kleinste", "laagste", "geringste", "bijna niet",
    "praktisch", "feitelijk", "bruikbaar", "redelijk", "redelijke", 
    "onredelijk", "onredelijke", "billijk", "billijke", "onbillijk", 
    "onbillijke", "terecht", "terechte", "onterecht", "onterechte",
    "logisch", "logische", "onlogisch", "onlogische", "rechtvaardig", "rechtvaardige",
    "onrechtvaardig", "onrechtvaardige", "slecht", "slechte", "zwak", 
    "zwakke", "gebrekkig", "gebrekkige", "beroerd", "beroerde", "waardeloos", 
    "waardeloze", "ietsje", "een fractie", "veel", "talrijk", "menig", 
    "een hoop", "vele", "talrijke", "talloze", "menige", "volkomen", 
    "onvolkomen", "geheel", "ongeheel", "helemaal", "onhelemaal", "totaal", 
    "ontotaal", "volledig", "onvolledig", "compleet", "oncompleet", "bijna", 
    "nagenoeg", "zo goed als", "haast", "niet", "geen", "nimmer", "nooit", 
    "geenszins", "niets", "niks", "geen enkel ding", "helemaal niets", "geen", 
    "niet één", "nul", "geeneen"]

deduplicated_words = list(set(words))

deduplicated_words.sort()

print(deduplicated_words)
