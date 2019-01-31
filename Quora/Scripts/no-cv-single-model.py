import gc
import re
import os
import pandas as pd
import numpy as np
import random
from sklearn import metrics
import string
import math
import operator
import time
from keras.preprocessing import text, sequence
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import psutil
from multiprocessing import Pool

change_string = '"single_model.py" : 20 fixed, 2 train, 5 patience, 2% validation, with adaptive lr patience=2 and no noise.\n'

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# GENERAL HYPERPARAMS
num_folds = 5
seed = 42

# HYPERPARAMS FOR TEXT PROCESSING
max_features = 120000
maxlen = 100

# HYPERPARAMS FOR NN
batch_size = 1024
epochs_fixed = 20
epochs_trainable = 2
embed_size = 300
early_stopping_patience = 5
hidden_size = 60
lr_patience = 2

set_seed(seed)

# PATH TO DATA DIRECTORY
PATH = "../input/"

num_cores = psutil.cpu_count()
num_partitions = num_cores

def df_parallelize_run(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

# remove space
spaces = ['\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\x10', '\x7f', '\x9d', '\xad', '\xa0']
def remove_space(text):
    """
    remove extra spaces and ending space if any
    """
    for space in spaces:
        text = text.replace(space, ' ')
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    return text

# replace strange punctuations and raplace diacritics
from unicodedata import category, name, normalize

def remove_diacritics(s):
    return ''.join(c for c in normalize('NFKD', s.replace('ø', 'o').replace('Ø', 'O').replace('⁻', '-').replace('₋', '-'))
                  if category(c) != 'Mn')

special_punc_mappings = {"—": "-", "–": "-", "_": "-", '”': '"', "″": '"', '“': '"', '•': '.', '−': '-',
                         "’": "'", "‘": "'", "´": "'", "`": "'", '\u200b': ' ', '\xa0': ' ','،':'','„':'',
                         '…': ' ... ', '\ufeff': ''}
def clean_special_punctuations(text):
    for punc in special_punc_mappings:
        if punc in text:
            text = text.replace(punc, special_punc_mappings[punc])
    # 注意顺序，remove_diacritics放前面会导致 'don´t' 被处理为 'don t'
    text = remove_diacritics(text)
    return text

# clean numbers
def clean_number(text):
    if bool(re.search(r'\d', text)):
        text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)
        text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)
        text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)  
    return text

unknown_to_english_map = {'隶书': 'Lishu', 'तपस्': 'Tapse', 'ให้': 'give', '宗教官': 'Religious officer', '没有神一样的对手': 'No god-like opponent', 'יהודיות': 'Jewish women', '我明明是中国人': 'I am obviously Chinese.', 'お早う': 'Good morning', 'मेरे': 'my', 'পরীক্ষা': 'Test', 'נעכטן': 'Yesterday', '中华民国': 'Republic of China', 'तकनीकी': 'Technical', 'यहाँ': 'here', '奇兵隊': 'Qibing', 'かﾟ': 'Or', '知识分子': 'Intellectuals', 'ごめなさい': "I'm sorry", '하기를': 'To', 'गाढवाचे': 'Donkey', 'इबादत': 'Worship', '千金': 'Thousand gold', 'ельзи': 'elzhi', '低端人口': 'Low-end population', '맛저': 'Mazur', 'दानि': 'Danias', 'юродство': 'foolishness', 'τον': 'him', '素质': 'Quality', '王晓菲': 'Wang Xiaofei', 'माहेर': 'Maher', 'لمصارى': 'For my advice', '送客！': 'see a visitor out!', '然而': 'however', 'जिग्यासा': 'Jigyaasa', '都市': 'city', 'ஜோடி': 'Pair', 'لاني': 'See', 'пельмени': 'dumplings', '请不要误会': "Please don't misunderstand", 'люис': 'Lewis', '不送！': 'Do not send!', 'के': 'Of', 'मस्का': 'Mascara', 'вoyѕ': 'voyce', 'बुन्ना': 'Bunna', '战略支援部队': 'Strategic support force', '平埔族': 'Pingpu', 'فهمت؟': 'I see?', 'कयामत': 'Apocalypse', 'ức': 'memory', 'ᗰeᑎ': 'ᗰe ᑎ', 'सेकना': 'Sexting', 'धकेलना': 'Shove', 'পারি': 'We can', 'مقام': 'Official', 'बेसति': 'Baseless', '歪果仁研究协会': 'Hazelnut Research Association', '短信': 'SMS', 'всегда': 'is always', '修身': 'Slim fit', 'إنَّ': 'that', '不是民国': 'Not the Republic of China', '写的好': 'Written well', 'õhtupoolik': 'afternoon', 'तालीन': 'Training', 'और': 'And', 'щит': 'shield', 'ᗩtteᑎtiᐯe': 'ᗩtte ᑎ ti ᐯ e', 'اسکوپیه': 'Script', 'çomar': 'fido', '和製英語': 'Japanglish', '吊着命': 'Hanging', 'много': 'much', 'समुराय': 'Samurai', 'भटकना': 'Wander', 'مقاومة': 'resistance', '싱관없어': 'I have no one.', '修身養性': 'Self-cultivation', 'मटका': 'pot', 'θιοψʼ': 'thiós', 'ㅎㅎㅎㅎ': 'Hehe', 'تساوى': 'Equal', 'बाट': 'From', '不地道啊。': 'Not authentic.', 'контакт': 'contact', '이런것도': 'This', 'तै': 'The', 'मेल': 'similarity', 'álvarez': 'Alvarez', 'नुक्स': 'Damage', '口訣': 'Mouth', 'масло': 'butter', 'परम्परा': 'Tradition', '学会': 'learn', 'کردن': 'Make up', 'öd': 'bile', 'टशन': 'Tashan', 'つらやましす': 'I am cheating', 'чего': 'what', '为什么总有人能在shoplex里免费获取iphone等重金商品？': 'Why do people always get free access to iphone and other heavy commodities in shoplex?', 'श्री': 'Mr', 'प्रेषक': 'Sender', 'خواندن': 'Read', 'बदलेंगे': 'Will change', 'बीएड': 'Breed', 'अदा': 'Paid', 'फैलाना': 'spread', 'ബുദ്ധിമാനായ': 'Intelligent', '谢谢六佬': 'Thank you, Liu Wei', 'উপস্থাপন': 'Present', 'بكلها': 'All of them', 'जाए': 'Go', '无可挑剔': 'Impeccable', 'आना': 'come', '太阴': 'lunar', 'القط': 'The cat', '있네': 'It is.', 'कर्म': 'Karma', 'आड़': 'Shroud', 'įit': 'IIT', 'जशने': 'Jashnay', 'से': 'From', 'åkesson': 'Åkesson', '乳名': 'Milk name', '我看起来像韩国人吗': 'Do I look like a Korean?', 'тнan': 'tn', 'العظام': 'Bones', '暹罗': 'Siam', '小肥羊鍋店': 'Little fat sheep pot shop', 'করেন': 'Do it', 'हरामी': 'bastard', 'डाले': 'Cast', 'استراحة': 'Break', '射道': 'Shooting', 'επιστήμη': 'science', 'χράομαι': "I'm glad", '脚踏车': 'bicycle', 'हिलना': 'To move', 'đa': 'multi', '標點符號': 'Punctuation', 'मैं': 'I', '不能化生水谷精微': "Can't transform the water valley", 'निवाला': 'Boar', 'दर्शनाभिलाषी': 'Spectacular', 'বাছাই': 'Picking', 'どくぜんてき': 'Dokedari', 'पीवें।': 'Pieve.', 'ʻoumuamua': 'Priority', 'में': 'In', 'चलाऊ': 'Run', 'निपटाना': 'To settle', 'ごはん': 'rice', 'चार्ज': 'Charge', 'शिथिल': 'Loosely', 'ястребиная': 'yastrebina', 'ложись': 'lie down', '李银河': 'Li Yinhe', 'へも': 'Also', 'हुलिया': 'Hulia', 'ऊब': 'Bored', 'छाछ': 'Buttermilk', 'عن': 'About', 'नहीं': 'No', 'जीव': 'creatures', 'जेहाद': 'Jehad', 'νερό': 'water', '열여섯': 'sixteen', 'मार': 'Kill', '早就': 'Long ago', '《齐天大圣》for': '"Qi Tian Da Sheng" for', 'الجنس': 'Sex', 'مع': 'With', 'रंगरलियाँ': 'Color palettes', 'जोल': 'Jol', 'बुद्धी': 'Intelligence', '罗思成': 'Luo Sicheng', '独善的': 'Self-righteousness', 'धर्मः': 'Religion', '中国大陆的同胞们': 'Chinese compatriots', '愛してない': 'I do not love you', '日本語が上手ですね': 'Japanese is good', 'ओठ': 'Lips', '如果你希望表達你的觀點': 'If you want to express your opinion', 'देशवाशी': 'Countryman', 'καί': 'and', '阮铭': 'Yu Ming', '跑步跑': 'Running', 'हो': 'Ho', '「褒められる」': '"Praised"', 'నా': 'My', '\x10new': '?new', 'ゆいがどくそん': 'A graduate student', '白语': 'White language', 'वह': 'She', '白い巨塔': 'White big tower', '로리이고': 'Lori.', 'परि': 'Circumcision', 'æj': 'Aw', 'انضيع': 'Lost', '平民苗字必称義務令': 'Civilian Miaozi must be called an obligation', 'ਸਿੰਘ': 'Singh', 'уже': 'already', 'đầu': 'head', '雨热同期': 'Rain and heat', 'घुलना': 'Dissolve', 'мис': 'Miss', 'て下さいませんか': 'Could it be?', '猫头鹰': 'owl', 'चढ़': 'Climbing', '漢音': 'Hanyin', 'यही': 'This only', '只有猪一样的队友': 'Only pig-like teammates', 
                          'übersoldaten': 'about soldiers', 'αθ': 'Ath', 'ουδεις': 'no', '党国合一': 'Party and country', 'रेड': 'Red', 'ढोलना': 'Drift', 'शाक': 'Shake', '같이': 'together', '攻克': 'capture', 'özalan': 'exclusive', 'काम': 'work', 'चाहना': 'Wish', '坚持一个中国原则': 'Adhere to the one-China principle', '배우고': 'Learn', '柴棍': 'Firewood stick', 'उसे': 'To him', 'на': 'on', '無手': 'No hands', 'čvrsnica': 'Čvrsnica', 'ज़ार': 'Tsar', 'ˆo': 'O', '後宮甄嬛傳': 'Harem of the harem', '意音文字': 'Sound character', 'बहारा': 'Bahara', 'てみる': 'Try', '老铁': 'Old iron', '野比のび太': 'Nobita Nobita', 'याद': 'remember', 'پیشی': 'Surpass', 'توقعك': 'Your expectation', 'はたち': 'Slender', 'فزت': 'I won', '伊藤': 'Ito', 'कलेजे': 'Liver', 'αγεν': 'ruin', 'ìmprovement': 'Improvement', 'では': 'Then.', 'பார்ப்பது': 'Viewing', '只留清气满乾坤': 'Only stay clear and full of energy', '이정도쯤': 'About this', 'すてん': 'Sponge', 'ما': 'What', '海南人の日本': "Hainan people's Japan", '小鹿乱撞': 'very excited', 'ῥωμαῖοι': 'ῥomaşii', 'वारी': 'Vary', 'भोज': 'Feast', '陈庚': 'Chen Geng', 'कट्टरवादि': 'Fanatic', '凌霸': 'Lingba', 'पार': 'The cross', 'कुचलना': 'To crush', 'रहा': 'Stayed', 'हम': 'We', 'бойся': 'be afraid', '大刀': 'Large sword', 'ন্ন': 'Done', '汽车': 'car', 'কে': 'Who', 'الکعبه': 'Alkaebe', '网络安全法': 'Cybersecurity law', 'नेपाली': 'Nepali', 'পদের': 'Position', '\x92t': '??t', 'रास': 'Ras', 'لكل': 'for every', 'शूद्राणां': 'Shudran', '此问题只是针对外国友人的一次统计': 'This question is only a statistic for foreign friends.', 'ឃើញឯកសារអំពីប្រវត្តិស្ត្រនៃប្រាសាទអង្គរវត្ត': 'See the history of Angkor Wat', 'спасибо': 'thank', 'رب': 'God', 'वजूद': 'Non existent', 'पकड़': 'Hold', 'बरहा': 'Baraha', '雾草': 'Fog grass', 'мощный': 'powerful', 'বৃদ্ধাশ্রম': 'Old age', 'आई': 'Mother', 'खड़ी': 'Steep', '\x1aaùõ\x8d': '? aùõ ??', 'āto': 'standard', '在冰箱裡': 'in the refrigerator', 'đời': 'life', 'लड़का': 'The boy', 'τὸ': 'the', 'مدرسان': 'Instructors', 'பறை': 'Drum', 'índia': 'India', 'दिया': 'Gave', '小篆': 'Xiao Yan', '唐樂': 'Tang Le', '米国': 'USA', '文言文': 'Classical Chinese', 'उवाच': 'Uwach', 'هدف': 'Target', 'घेरा': 'Cordon', 'झूम': 'Zoom', 'εσσα': 'all the same', '和製漢語': 'And making Chinese', 'ऐलोपैथिक': 'Allopathic', 'создала': 'has created', 'إدمان': 'addiction', '游泳游': 'Swimming tour', '狮子头': 'Lion head', 'दिये': 'Given', 'पास': 'near', 'εντjα': 'in', 'まする': 'To worship', 'हाल': 'condition', '짱깨': 'Chin', 'জল': 'Water', 'चीज': 'thing', 'îmwe': 'one', '胡江南': 'Hu Jiangnan', 'तमाम': 'All', 'कट्टर': 'Hardcore', '魔法师': 'Magician', 'బుూ\u200c': 'Buu', 'चनचल': 'Chanchal', 'सीधा': 'Straightforward', '不要人夸颜色好': "Don't exaggerate the color", 'हिमालय': 'Himalaya', 'सिंह': 'Lion', 'خراميشو': 'Wastewater', 'мoѕт': 'the moment', 'কোথায়': 'Where?', 'نصف': 'Half', 'رائع': 'Wonderful', 'добрый': 'kind', 'ज़र्रा': 'Zarra', 'يعقوب': 'Yacoub', 'सम्झौता': 'Agreement', 'ān': 'yes', '했다': 'did', 'ἀριστοκράτης': 'aristocrat', 'çan': 'Bell', '太阳': 'sun', 'सर': 'head', 'गुरु': 'Master', '嘉定': 'Jiading', '故乡': 'home', '安倍晋三': 'Shinzo Abe', 'मच्छर': 'Mosquito', 'सभी': 'All', '成語': 'idiom', '唯我独尊': 'Only me', 'นะค่ะ': 'Yes', 'นั่นคุณกำลังทำอะไร': 'What are you doing', 'ın': 's', 'الشامبيونزليج': 'Shamballons', '抗日神剧': 'Anti-Japanese drama', '小妹': 'Little sister', 'çok': 'very', 'लोड': 'Load', 'साहित्यिक': 'Literary', 'लाल।': 'Red.', 'فى': 'in a', '絕不放過一人」之勢': 'Never let go of one person', '国家主席': 'National president', '异议': 'objection', 'अनुस्वार': 'Anuswasar', 'តើបងប្អូនមានមធ្យបាយអ្វីខ្លះដើម្បីរក': 'What are some ways to find out', '允≱ၑご搢': '≱ ≱ ≱ 搢', '黒髪': 'Black hair', '自戳双目': 'Self-poke binocular', 'βροχέως': 'rainy', 'ని': 'The', '一带一路': 'Belt and Road', 'śliwińska': 'Śliwińska', 'κατὰ': 'against', '打古人名': 'Hit the ancient name', 'குறை': 'Low', 'üjin': 'Uji', 'öppning': 'opening', 'अन्जल': 'Anjal', '이와': 'And', 'आविष्कार': 'Invention', 'غنج': 'Gnostic', 'рассвет': 'dawn', 'होना': 'Happen', 'বাড়ি': 'Home', '入定': 'Enter', 'भेड़': 'The sheep', 'देगा': 'Will give', 'ब्रह्मा': 'Brahma', 'बीन': 'Bean', 'χράω': "I'm glad", 'तीन': 'three', 'ضرار': 'Drar', 'विराजमान': 'Seated', 'सैलाब': 'Salab', 'συνηθείας': 'communication', '구경하고': 'To see', 'चुल्ल': 'Chool', '红宝书': 'Red book', '羡慕和嫉妒是不一样的。': 'Envy and jealousy are not the same.', 'मेरा': 'my', 'कलटि': 'Katiti', 'हिमाती': 'Himalayan', 'ಸಾಧು': 'Sadhu', 'عطيتها': 'Her gift', 'छान': 'Nice', 'เกาหลี': 'Korea', 'íntruments': 'instruments', 'يتحمل': 'Bear', 'रुस्तम': 'Rustam', 'बरात': 'Baraat', 'रंग': 'colour', '나이': 'age', 'פרפר': 'butterfly', '老乡': 'Hometown', '謝謝': 'Thank you', 'í‰lbƒ': '‰ in lbƒ', 'हरजाई': 'Harjai', 'পেতে': 'Get to', '「勉強': '"study', 'रड़कना': 'To cry', '清真诊所': 'Halal clinic', 'أفضل': 'Best', 'استطيع': 'I can', 'नाकतोडा': 'Nag', 'बड़ि': 'Elder', 'वैद्य': 'Vaidya',
                          'أنَّ': 'that', 'いたロリィis': 'There was Lory', 'ìn': 'print', '本人堅持一個中國的原則': 'I adhere to the one-China principle', 'океан': 'ocean', 'बाद': 'after', '禮儀': 'etiquette', 'सिषेविरे': 'Sisavir', 'अमृत': 'Honeydew', 'بدو': 'run', 'ὅς': 'that is', 'छोटा': 'small', 'स्वभावप्रभवैर्गुणै': 'Nature is bad', 'ענג': 'Tight', 'שלמה': 'Complete', 'डोरे': 'Dore', '我害怕得毛骨悚然': 'I am afraid of creeps.', 'झोलि': 'Jolly', 'единадесет': 'eleven', 'என்ன': 'What', 'आयुरवेद': 'Ayurveda', '堵天下悠悠之口': 'Blocking the mouth of the world', 'الجمعي': 'Collective', 'ángel': 'Angel', 'रोना': 'crying', 'हमराज़': 'Humraj', 'चिलमन': 'Drape', 'औषध': 'Medicine', 'निया': 'Nia', 'תעליעוכופור': 'Go up and go', '中国队大胜': 'Chinese team wins', 'δ##': 'd ##', 'चला': 'walked', 'पर': 'On', 'ἔργον': 'work', 'रंक': 'Rank', 'सिक्ताओ': 'Sixtao', '陳太宗': 'Chen Taizong', 'لومڤور': 'لوموور', 'விளையாட்டு': 'Sports', '的？': 'of?', '안녕하세요': 'Hi', 'נבאים': 'Prophets', 'ন্য': 'N.', 'şimdi': 'now', 'भरना': 'Fill', 'धरी': 'Clog', '漢字': 'Chinese character', 'इसि': 'This', 'आवर': 'Hour', 'काटना': 'cutting', '僕だけがいない街': 'City where I alone does not exist', 'जूठा': 'Lier', 'çekerdik': 'We Are', 'čaj': 'tea', 'నందొ': 'Nando', '核对无误': 'Check is correct', '無限': 'unlimited', 'समेटना': 'Crimp', 'żurek': 'sour soup', 'আছে': 'There are', 'مرتك': 'Committed', 'आपाद': 'Exorcism', 'चोरी': 'theft', 'బవిష్కతి': 'Baviskati', 'निकालना': 'removal', 'सीमा': 'Limit', '配信開始日': 'Distribution start date', '宝血': 'Blood', 'ग्यारह': 'Eleven', 'गरीब': 'poor', '있고': 'There', 'île': 'island', 'स्मि': 'Smile', 'енгел': 'engel', 'вы': 'you', 'οπότε': 'so', 'सूनापन': 'Desolation', 'áraszt': 'exudes', 'मारि': 'Marie', '서로를': 'To each other', 'घोपना': 'To announce', 'फितूर': 'Fitur', 'あからさまに機嫌悪いオーラ出してんのに話しかけてくるし': 'I will speak to the out-putting aura outright bad', '封柏荣': 'Feng Bairong', '「褒められたものではない」': '"It is not a praise"', '傳統文化已經失存了': 'Traditional culture has lost', '水木清华': 'Shuimu Tsinghua', 'पन्नी': 'Foil', 'ਰਾਜਬੀਰ': 'Rajbir', '煮立て': 'Boiling', '外国的月亮比较远': 'Foreign moon is far away', 'شغف': 'passion', 'గురించి': 'About', 'övp': 'ÖVP', '实名买票制': 'Real name ticket system', 'बिखरना': 'To scatter', 'التَعَمّقِ': 'Persecution', 'जीवें': 'The living', 'गई': 'Has been', '部頭合約': 'Head contract', 'دقیانوس': 'Precise', 'फंकी': 'Funky', '粵拼': 'Cantonese fight', 'ĉiohelpanto': 'all supporter', 'मारना': 'killing', 'मानजा': 'Manja', 'अष्टांगिक': 'Octagonal', 'への': 'To', 'रहना': 'live', 'खाना': 'food', 'ಕೋಕಿಲ': 'Kokila', 'చిన్న': 'Small', '金髪': 'Blond hair', '籍贯': 'Birthplace', 'उखाड़ना': 'Extirpate', 'α2': "A'2", 'להשתלט': 'take over control', '露西亚': 'Lucia', '大有「寧可錯殺一千': 'There is a lot of "I would rather kill a thousand', 'संस्कृत': 'Sanskrit', 'христо': 'Christo', 'लगाना': 'to set', '선배님': 'Seniors', 'उड़ीसा': 'Orissa', 'ஆய்த': 'Ayta', 'तेरे': 'Your', 'आकांक्षी': 'Aspiring', 'बाज़ार': 'The market', 'हर्ज़': 'Herz', 'だします': 'Will do.', 'चक्कर': 'affair', '없어': 'no', 'चम्पक': 'Champak', 'ताल': 'rythm', 'āgamas': 'hobby', 'योद्धा': 'Warrior', 'αλυπος': 'chain', 'बेड़ा': 'Fleet', 'बात': 'talk', '做莫你酱紫的？': 'Do you make your sauce purple?', '見逃し': 'Miss', '鰹節': 'Bonito', 'तथैव': 'Fact', 'आध्यात्मिकता': 'Spirituality', '正弦': 'Sine', 'लिया': 'took', 'îndrăgire': 'love', '我願意傾聽': 'I am willing to listen', '家乡': 'hometown', '大败': 'Big defeat', 'मए': 'Ma', 'జ్ఞ\u200cా': 'Sign', 'օօ': 'oo', 'खेमे': 'Camps', 'الشوكولاه': 'Chocolate', 'полностью': 'completely', '商品発売日': 'Product Release Date', '金継ぎ': 'Gold piecing', '烤全羊是多少人民币呢？': 'How much is the renminbi roasted in the whole sheep?', 'γолемата': 'golem', '孫瀛枚': 'Grandson', 'özlüyorum': 'I am missing', 'خلبوص': 'Libs', 'アンダージー': 'Undergar', '蝴蝶': 'butterfly', 'প্রশ্ন': 'The question', 'जरूरी': 'Necessary', 'बूरा': 'Bura', '有毒': 'poisonous', 'सान्त्वना': 'Comfort', '눈치': 'tact', '\ufeffwhat': 'what', 'çeşme': 'fountain', 'ごっめんなさい': 'Please, sorry.', 'விளக்கம்': 'Description', '罗西亚': 'Rosia', 'गोटा': 'Knot', 'लेखावलिपुस्तके': 'Accounting Tips', 'सम्भोग': 'Sexual intercourse', '慢走': 'Slow walking', 'तुम': 'you', 'लीला': 'Leela', 'ฉันจะทำให้ดีที่สุด': 'I will do my best', 'चम्पु': 'Shampoo', '視聽華語': 'Audiovisual Chinese', '比比皆是': 'Abound', 'धावा': 'Run', '娘娘': 'Goddess', 'पहाड़': 'the mountain', 'राजा': 'King', '茹西亚': 'Rusia', 'ब्राह्मणक्षत्रियविशां': 'Brahmin Constellations', '수업하니까': "I'm in class.", 'చెల్లెలు': 'Sister', 'साजे': 'Made', '支那人': 'Zhina', '团表': 'Group table', '讲得': 'Speak', 'へからねでて': 'From to', 'über': 'over', 'քʀɛʋɛռt': 'kʀɛʋɛr t', 'นเรศวร': 'Naresuan', '方言': 'dialect', 'पना': 'Find out', '怕樱': 'Afraid of cherry', 'घुल': 'Dissolve', '米帝': 'Midi', 'طيب': 'Ok', 'प्रेम': 'Love', 'पढ़ाई': 'studies', '他穿褲子': 'He wears pants', 'मेरी': 'Mine',
                          'উত্তর': 'Reply', 'स्थिति': 'Event', '\x1b\xadü': '?\xadü', 'तैश': 'Tachsh', '写得好': 'Well written', 'मॉलूम': 'Known', '创意梦工坊': 'Creative Dream Workshop', 'आपकी': 'your', 'मिलना': 'Get', '배웠어요': 'I learned it', '許自東': 'Xu Zidong', 'जाऊँ': 'Go', 'अहिंसा': 'Nonviolence', 'джоу': 'Joe', '金繕い': 'Gold patting', '好心没好报': 'No return on a good deed', 'çevapsiz': 'unanswered', 'मिल': 'The mill', '日本語': 'Japanese', 'اخذ': 'Get', '水军': 'Water army', 'बिना': 'without', 'बनाना': 'Make', 'التوبه': 'Repentance', '一个灵运行在我的家': 'a spirit running in my home', '한국어를': 'Korean', 'कि': 'That', 'εισα': 'import', 'लगना': 'feel', 'गपोड़िया': 'Gopodiya', 'ārūpyadhātu': 'exhyadadate', 'आए': 'Returns', '吃好吃金': 'Eat delicious gold', 'นั่น': 'that', 'बढ़ाना': 'raise up', '보니': 'Bonnie', '爱着': 'Love', '선배': 'Elder', '刷屏': 'Brush screen', '人性化': 'Humanize', 'خرسانة': 'Concrete', '麻辣乾鍋': 'Spicy dry pot', '\x10œø': '? œø', 'सतरंगी': 'Satarangi', '磨合': 'Run-in', 'को': 'To', 'شهادة': 'Degree', 'একটি': 'A', 'へと': 'To', 'يلي': 'Following', '光本': 'Light source', '褒め殺し': 'Praise kill', 'आँचल': 'Anchal', 'এর': 'Of it', 'पिटा': 'Pita', 'بديش': 'Badish', 'गंगु': 'Gangu', '可是': 'but', 'की': 'Of', '谢谢。台灣同胞': 'Thank you. Taiwan compatriots', 'दम': 'power', 'मैकशि': 'Macshi', 'రాజ': 'King', '玉蘭花': 'Magnolia', '江ノ島盾子': 'Enoshima Junko', 'ѕтυpιd': 'ѕtυpιd', 'जिसका': 'Whose', 'எழுத்து': 'Letter', '甲骨文': 'Oracle', 'चूरमा': 'Churma', 'चूलें': 'Chulen', 'प्रविभक्तानि': 'Interpretation', 'いる': 'To have', 'مقال': 'article', 'पाय': 'Feet', 'अतीत': 'Past', 'ármin': 'Armin', '東夷': 'Dongyi', 'आदमकद': 'Life expectancy', 'किये': 'Done', 'पतवार': 'Helm', '楽曲派アイドル': 'Song musical idols', 'डगमगाना': 'Waver', '북한': 'North Korea', '禮記': 'Book of Rites', '西魏': 'Xi Wei', '過労死': 'Death from overwork', 'बेबुनियाद': 'Unbounded', '仙人跳': 'Fairy jump', '港女': 'Hong Kong girl', '虞朝': 'Sui Dynasty', 'µ0': 'μ0', '字母词': 'Letter word', 'अर्धांगिनि': 'Arghangini', '真功夫': 'real Kong Fu', '飯糰': 'Rice ball', 'علم': 'Science', 'गुड़': 'Jaggery', 'гречку': 'buckwheat', '我方记账数字与贵方一致': 'Our billing figures are consistent with yours', 'आने': 'To arrive', 'कब': 'When', '宋楚瑜': 'James Soong', 'διητ': 'filter', 'राई': 'Rye', 'விரதம்': 'Fasting', 'बदल': 'change', '成语': 'idiom', 'ĺj': 'junk', 'какая': 'which one', '文翰': 'Wen Han', 'کـ': 'K', 'ʀɛċօʍʍɛռɖɛɖ': 'ʀɛċ oʍʍɛрɖɛɖ', 'दिलों': 'Hearts', '星期七': 'Sunday', '傳送': 'Transfer', '陳云根': 'Chen Yunen', 'ʿalaʾ': 'Allah', 'गुल': 'Gul', 'ख़बर': 'The news', 'मन्च': 'Manch', '国家知识产权局': 'State Intellectual Property Office', '행복하게': 'happily', 'बबीता': 'Babita', 'юродивый': 'holy fool', 'सफाई': 'clean', 'вода': 'water', 'लाठि': 'End', 'γὰρ': 'γσρ', '在日朝鮮人／韓国人': 'Koreans/Koreans in Japan', '学过': 'Learned', 'जमाना': 'Solidification', 'इकोनॉमिक्स': 'Economics', 'क्या': 'what', 'ドア': 'door', '中国的存在本身就是原罪': 'The existence of China itself is the original sin', 'मौसमि': 'Season', 'ठाठ': 'Chic', 'ἡμετέραν': 'another', '火了mean': 'Fired mean', 'せえの': 'Set out', 'ஆயுத': 'Arms', 'कंपनी': 'company', 'अहम्': 'Ego', 'भर': 'Filled', 'फिरना': 'To revolve', '高山族': 'Gaoshan', '王飞飞是一个哑巴的婊子。': 'Wang Feifei is a dumb nephew.', '反清復明': 'Anti-clearing', '关门放狗': 'Close the dog', '中国民族伟大复兴？i': 'The great revival of the Chinese nation? i', '漢名': 'Han name', 'ín': 'into the', '이름은': 'name is', '뽀비엠퍼러': 'Pobi Emperor', '堅決反對台獨言論': 'Resolutely oppose Taiwan independence speech', 'ड़': 'D', 'ادب': 'Literary', '〖2x〗': '〖〗 2x', 'في': 'in a', 'परन्तप': 'Parantap', '不正常人類研究中心': 'Abnormal human research center', 'метью': 'by the way', 'σε': 'in', '罗马炮架': 'Roman gun mount', 'đỗ': 'parked', '二哈': 'Erha', '中國': 'China', 'मुझे': 'me', 'бджа': 'bzha', 'छुटाना': 'To leave', 'সহ': 'Including', 'δίδυμος': 'twin', 'दौरा': 'Tour', 'आया': 'He Came', 'ľor': 'Lor', 'شاف': 'balmy', 'افشلك': 'I miss you', 'पता': 'Address', 'śląska': 'Silesian', '我还有几个条件呢。': 'I still have a few conditions.', '元寇': 'Yuan', 'सहायक': 'Assistant', 'टाइम': 'Time', 'समुदाय': 'Community', 'टिपटिपवा': 'Tiptipava', '五毛党': 'Wu Mao Party', 'दे': 'Give', '屠城': 'Slaughter city', 'कहने': 'To say', 'कलमूहा': 'Kalamuha', 'בפנים': 'in', 'कर्माणि': 'Creation', 'अथवा': 'Or', 'રાજ્ય': 'State', 'घसोना': 'Shed', '今そう言う冗談で笑える気分じゃないから一人にしてって言ったら何があったの？話聞くよって言われたんだけど': 'I do not feel like being laughing with what I say now, so what happened when I told you to be alone? I was told by listening and talking', 'परमो': 'Paramo', 'υφηιρειτω': 'I suppose', 'כתובים': 'Written', '能夠': 'were able', '고등학교는': 'High School', 'बजरि': 'Breeze', 'जानेदिल': 'Knowingly', '大胜': 'Big win', 'काल': 'period', 'すみません': "I'm sorry", 'हिमाकत': 'Snow plant',
                          'þîfû': 'iphone', 'よね': 'right', 'هالنحس': 'Halting', 'الحجازية': 'Hijaz', 'жизнь': 'a life', 'किया': 'Did', '沸騰する': 'To boil', 'นะครับ': 'Yes', 'پیش': 'Before', 'परदाफास': 'Bustard', 'वफाएं': 'Affection', 'सैनानि': 'Sanani', 'ölü': 'dead', 'हैं': 'Are', '宋美齡': 'Song Meiling', 'கவாடம்': 'Valve', '我是一名来自大陆的高中生': 'I am a high school student from the mainland.', 'اجمل': 'The most beautiful', '가용': 'Available', 'चरम': 'Extreme', '象形文字': 'Hieroglyphics', '男女授受不亲': 'Men and women don’t kiss', 'моя': 'my', '疑义': 'Doubt', '管中閔': 'In the tube', 'çekseydin': 'pulls you were', 'عم': 'uncle', 'ルージュ': 'Rouge', '永遠': 'forever and always', 'بيت': 'a house', '「褒められた」': '"I was praised"', 'कर्णजित्': 'Karnajit', 'あたまわるい': 'Bad headache', 'ε0': 'e0', 'şafak': 'dawn', 'जलाओ': 'Burn', '観世音菩薩': '観世音', 'पीत': 'Yellow', 'दारी': 'Dari', 'የሚያየኝን': 'What i see', 'هاپو': 'Dog', '发声点': 'Vocal point', 'être': 'be', 'மாண்பாளன்': 'Manpalan', '감겨드려유': 'You can wrap it.', '知音': 'Bosom friend', 'एक': 'One', '\x01jhó': '? Jho', 'かんぜおんぼさつ': 'Punctuation', 'ødegaard': 'Ødegaard', '得理也要让人': 'It’s ok to make people', 'مرة': 'Once', 'दो': 'two', 'ने': 'has', '用乡村包围城市': 'Surround the city with the countryside', 'مكتب': 'Office', 'řepa': 'beet', 'день': 'day', '江青': 'Jiang Qing', 'ángeles': 'angels', 'çonstant': 'constant', 'टिपटिपुआ': 'Tip Tip', 'ஆன்லைன்': 'Online', '盧麗安': "Lu Li'an", 'самый': 'most', 'からかってくる男に': 'To the coming men', 'بعرف': 'I know', '知乎': 'Know almost', 'पहेलि': 'Puzzles', 'बहार': 'spring', 'भंगिमाँ': 'Bhangima', 'くわんぜおんぼさつ': 'Honey', '河殤': 'River', 'شو': 'Shaw', 'محمد': 'Mohammed', 'спереди': 'in front', '没毛病': 'No problem', 'árbenz': 'Arbenz', 'लार्वा': 'Larvae', 'चढ़ा': 'Ascend', 'तो': 'so', 'يلعب': 'Play', 'घिसा': 'Ginger', 'रेखागड़ित': 'Sketchy', 'मुरदा': 'Murada', 'चित्र': 'picture', 'தாக்கல்': 'Filing', '大败美国队': 'Big defeat to the US team', 'искусственный': 'artificial', 'चाल': 'Trick', 'घुटता': 'Kneeling', 'बिगड़ना': 'Deteriorate', '您这口味奇特也就罢了': 'Your taste is strange.', 'लायक': 'Worth', '大篆': 'Daxie', 'खिलाना': 'To feed', 'ماهو': 'What is the', 'पहुँचनेके': 'To reach', '外来語': 'Foreign language', 'चटका': 'Click', 'کوالا': 'Koala bear', '不过': 'but', 'पड़ना': 'Fall', 'पानी': 'Water', '저의': 'my', '一生懸命': 'Hard', 'مش': 'not', 'लिखामि': 'Written', 'गया': 'Gaya', 'компания': 'company', 'तेरि': 'Teri', '茶髪': 'Brown hair', '하다': 'Do', '不是共和国；中华人民共和国和朝鲜民主主义人民共和国是共和国': 'Not a republic; the People’s Republic of China and the Democratic People’s Republic of Korea are Republic', '烤全羊多少人民币呢？': 'How much is the price of roast whole sheep?', 'घर': 'Home', 'पाला': 'Frost', '左右手': 'Left and right hand', 'کوالالامپور': 'Kuala Lumpur', 'āsanas': 'asanas', 'ոո': 'e', 'сегодня': 'Today', '저는': 'I am', 'русофил': 'blossoming', '甘蓝': 'Cabbage', '流浪到淡水': 'Wandering to fresh water', 'единайсет': 'eleven', 'परात': 'Underwear', '\x7fhow': '?how', '¡que': 'what', 'ἀχιλλεύς': 'Achilles', 'так': 'So', 'آداب': 'Rituals', '\x10i': '?i', 'मट्ठा': 'Whey', '平天下悠悠之口': 'The mouth of the world', 'लुन्डा': 'Lunda', 'ся': 'camping', 'вареники': 'Vareniks', 'δο': 'gt;', 'öyle': 'so', 'पुरे': 'Enough', '看他不顺眼': 'Seeing him not pleasing to the eye', '「o」': '"O"', 'とする': 'To', 'হচ্ছে': 'Being', 'लाखों': 'Millions', 'α1': "A'1", 'نهائي': 'Final', 'ɾ̃': 'ɾ', 'तिलक': 'Tilak', 'لا': 'No', 'صور': 'photo', '怒怼': 'Roar', 'மௌன': 'Mauna', 'परिस्थिति': 'Situation', '서로가': 'Mutually', '山進': 'Shanjin', '蝴蝶蛋': 'Butterfly egg', 'ксш': 'ksş', '饭可以乱吃': 'Rice can be eaten', '渡します': 'I will hand it over.', 'österreich': 'Austria', 'øÿ\x13': 'øÿ?', 'ممتاز': 'Excellent', '蝴蝶卵': 'Butterfly egg', 'ही': 'Only', 'ठोकरे': 'Knock', '湿婆': 'Shiva', 'обожаю': 'love', 'लस्सी': 'Lassi', '操你妈': 'Fuck your mother', 'फौलादि': 'Fauladi', 'жизни': 'of life', 'đổi': 'change', '阻天下悠悠之口': 'Block the mouth of the world', 'オメエだよオメエと話してんのが苦痛なんだよ。シネ。': "It's omn it is painful to talk to Oume. Cine.", 'खाट': 'The cot', '译文': 'Translation', 'सँवरना': 'To embellish', 'дванадесет': 'twelve', '陳雲': 'Chen Yun', 'дванайсет': 'twelve', 'आँखें': 'Eyes', 'पूछते': 'Inquires', 'भंगी': 'Posture', 'सूना': 'Deserted', 'प्याला': 'Cup', 'には': 'To', 'доктора': 'the doctors', 'देना': 'give', 'बिरेन्र्द': 'Forget', 'गला': 'throat', 'रखा': 'Kept', 'हाले': 'Haley', '欢迎入坑': 'Welcome to the pit', 'डलि': 'Dallie', 'ôš': 'OS', 'يوسف': 'Yousuf', 'छोंकना': 'Strain', 'пп': 'pp', 'उसपर': 'on that', 'υρολογιστών': 'computers', '新年快乐！学业进步！身体健康！谢谢您们读我的翻译篇章': 'happy New Year! Academic progress! Healthy body! Thank you for reading my translation chapter.', '人民': 'people', 'घंटे': 'Hours', 'شباط': 'February', '食べる': 'eat',
                          'صلاح': 'Salah', '土澳': 'Tuao', '干嘛天天跟我说韩语': 'Why do you speak Korean with me every day?', 'ōnogi': 'climate', 'صاحب': 'owner', 'اكل': 'ate', '大唐': 'Datang', 'مطالعه': 'Study', '养生': 'Health', '车子': 'Car', 'कचरा': 'Garbage', 'महरबानि': 'Mehraban', 'शुष्टि': 'Shutti', 'интеллект': 'intelligence', '阮鏐': '阮镠', '鲁玥': 'Reckless', '입니다': 'is', 'ῥιζὤματα': 'Threads', 'люблю': 'love', 'ɛxɛʀċɨsɛ': 'it is not', 'कपड़े': 'dresses', 'उड़न': 'Flying', '广电总局': 'SARFT', '骂人': 'curse', 'የየየኝን': 'What do i say', 'बेलना': 'Crib', 'பல்லாக்குப்': 'Pallakkup', 'बराबर': 'equal', 'ظرف': 'Dish', 'होनोपैथिक': 'Homeopathic', '君子': 'Gentleman', '河和湖': 'Kawahata lake', '精一杯': 'Utmost', 'है': 'is', '非要以此为依据对人家批判一番': 'I have to criticize others on this basis.', 'வச்சன்': 'Vaccan', 'நான்': 'I', 'का': 'Of', '三味線': 'Shamisen', 'šwejk': 'švejk', 'дурак': 'fool', '风琴': 'organ', 'हिंसा': 'Violence', 'βιον': 'bio', 'नारी': 'Woman', '知らない': 'Do not know', 'मुँह': 'The mouth', 'अपरम्पार': 'Unperturbed', '秋季新款': 'Autumn new style', 'আমার': 'Me', 'عبقري': 'genius', 'आहें': 'Ah', 'ਨਾਮ': 'Name', 'महफ़िल': 'Mehfil', 'बटेर': 'Quail', '林彪': 'Lin Wei', 'जाने': 'Know', 'डोंगरिचाल': 'Mountain move', 'εἰρήνη': 'Irene', 'प्रतिलेखनम्': 'Transcript', 'дп': 'dp', 'उसने': 'He', '몇시간': 'how many hours', 'नैया': 'Naiya', 'ἐξήλλακτο': 'inexpensive', '彩蛋': 'Egg', 'उलटफेर': 'Reverse', '台湾最美的风景是人': 'The most beautiful scenery in Taiwan is people.', '馄饨': 'ravioli', 'सदके': 'Shake', '饺子': 'Dumplings', 'भूमिका': 'role', '为什么说': 'Why do you say', '요즘': 'Nowadays', 'गिरि': 'Giri', '中國話': 'Chinese words', 'द्रोहि': 'Drohi', '中庸之道': 'The doctrine of the mean', 'хочу': 'want', 'ḵarasāna': 'ḵarasana', '走gǒ': 'Go gǒ', 'जमाल': 'Jamal', 'मन': 'The mind', 'तेलि': 'Oilseed', '非诚勿扰': 'You Are the One', 'атом': 'atom', '中华民国和大韩民国是民国': 'The Republic of China and the Republic of Korea are the Republic of China', 'नलि': 'Nile', 'हाथ': 'hand', 'खाजला': 'Itching', 'ōe': 'yes', '것이다': 'will be', 'şoųl': 'şoùl', 'तश्तरि': 'Cleverness', 'χρῆσιν': 'use', '民族': 'Nationality', 'الرياضيات': 'Mathematics', 'にじゅうさい': 'Twelve months', 'ὠς': 'as', '恋に落ちないからよく悲しい': 'It is often sad because it does not fall in love', 'ਸ਼ੀਂਹ': 'Lion', '黎氏玉英': 'Li Shiyuying', 'فبراير': 'February', '白濮': 'Chalk', 'অধীনে': 'Under', 'प्रशंसा': 'appreciation', 'ขอพระเจ้าอยู่ด้วย': 'May God be with you', 'अडला': 'Bent', 'ده': 'Ten', 'पसीजना': 'Exudate', 'कोई': 'someone', 'கருக்குமட்டை': 'Karukkumattai', 'कन्नी': 'Kanni', 'यात्रा': 'journey', '白酒': 'Liquor', 'τὴν': 't', 'करना': 'do', 'उल': 'ul', '俄罗斯': 'Russia', 'உனக்கு': 'You', 'जौहर': 'Johar', 'ಸ್ವರಕ್ಷರಗಳು': 'Self-defense', '论语': 'Analects', 'šakotis': 'branching', '儿臣惶恐': 'Childier fear', '讲的？': 'Said?', 'প্রতিনিয়ত': 'Every day', 'சன்னல்': 'Sill', '蠢的像猪一样': 'Stupid like a pig', 'رح': 'Please', 'بس': 'Yes', 'εὔιδον': 'you see', 'सदबुद्धि': 'Good sense', 'भगवान': 'God', '사랑해': 'I love you', 'בתוך': 'Inside', 'čeferin': 'чеферин', '民国': 'Republic of China', '但是': 'but', 'सकता': 'can', 'घाट': 'Wharf', 'čechy': 'Bohemia', '抹黑': 'Smear', 'γλαυκῶπις': 'greyhounds', 'నీకెందుకు': 'Nikenduku', 'चमत्कार': 'Miracle', 'दुनिया': 'world', 'یہاں': 'Here', 'اللي': 'Elly', 'খামার': 'The farm', '一呼百诺': 'One call', 'വിഡ്ഢി': 'Stupid', 'दिल': 'heart', 'тeenage': 'trinage', '皇上': 'emperor', 'टटोलना': 'Grope', '犬子': 'Dog', '我希望有一天你沒有公王病': 'I hope that one day you don’t have a king', '并无分裂中国的意图': 'No intention to split China', 'óscar': 'oscar', 'ኤልሮኢ': 'Alright', 'ŷhat': 'hhat', '천사': 'Angel', 'दी।': 'Given', 'रड़क': 'Raze', 'कानी': 'Kani', '江之島盾子': 'Enokima Shiko', '老生常谈': 'Old talk', 'των': 'of', '星期日': 'on Sunday', 'पैन्डा': 'Panda', 'マリも仲直りしました': 'Mari also made up.', 'ты': 'you', 'देश': 'Country', 'ठंडक': 'Coolness', 'নামল': 'Get down', 'जर्मनी': 'Germany', 'шли': 'walked', 'たべる': 'To eat', 'लिए': 'for', '鸡汤文': 'Chicken soup', 'มวยไทย': 'Thai boxing', '簡訊': 'Newsletter', 'منزل': 'Home', 'कर': 'Tax', '生女眞': 'Daughter-in-law', 'ек': 'ek', 'સંઘ': 'Union', '\u200bsalarpuria': 'Salphary', 'ţara': 'the country', 'नही': 'No', 'मगज': 'Mercury', 'অক্ষয়': 'Akshay', 'حال': 'Now', 'चिन्दी': 'Chindee', 'τῆς': 'her', '福哒柄': 'Good fortune', '청하': 'Qinghai', '越人': 'Yueren', 'なかなかに謎だな': "It's quite a mystery.", 'रखना': 'keep', 'பரை': 'Parai', 'करके': 'By doing', 'فِي': 'F', 'गुथना': 'Knit', '话不可以乱讲': "Can't talk nonsense", 'वैकल्पिक': 'Alternative', 'ਨਾਮੁ': 'Name', '你别高兴得太早': "Don't be too happy too early", '煎餅': 'Pancake', '한다': 'do', 'सबब': 'Cause', 'বিষয়টি': 'Matter', 'कोसना': 'To crack', 'ㅜㅜ': 'ㅜ', 'অক্সয়': 'Akshay', 'الدوالي': 'Varicose veins', 'பயிர்ப்பு': 'Yields', 'अजा': 'SC', 'あの色々': 'That kind of variety', 'емеля': 'emel', 'मेवा': 'Meva',
                          'जलवाफरोज़': 'Jalwa Phoroz', '中庸': 'Moderate', 'उसके': 'his', 'अहम': 'Important', 'वहम': 'Vanity', 'ís': 'ice', 'कलात्मक': 'Artistic', 'ἀχιλῆος': 'Achilles', '民族罪人': 'National sinner', 'मुकाम': 'Peer', 'ão': 'to', '한국': 'Korea', 'ادهر': 'Idir', '一長': 'One long', 'てくれませんか': 'Would you please', 'ブレイク': 'break', 'शहनाई': 'the clarinet', 'तीरन्दाज़ि': 'Arrows', 'रूढ़ीवादि': 'Conservative', 'झाँकी': 'Peeping', 'सत्यवादी': 'Truthful', '郑琳': 'Zheng Lin', 'युनानी': 'Unani', 'φώνας': 'light', 'जमना': 'Solidify', '〖plg〗': '〖Plg〗', 'हरी': 'Green', 'بطولة': 'championship', '这些词怎么读？这些词怎么说？这些词怎么念？which': 'How do you read these words? What do these words say? How do you read these words? Which', 'алтерман': 'alterman', 'اَلبَحْثِ': 'البحث', 'موجود': 'Available', 'कश्ति': 'Power', 'اسم': 'Noun', 'लाज़मि': 'Shameful', 'तुमने': 'you', '공화국': 'republic', 'लुक्का': 'Lukka', '史记': 'Historical record', '포경수술': 'Circumcised', '高端大气上档次into': 'High-end atmospheric grade into', 'что': 'what', 'योध': 'Rift', 'धर्म': 'religion', 'दरदर': 'Tariff', '訓読み': 'Kun Readings', 'впереди': 'in front', '민국': 'Republic of Korea', 'εντσα': 'in', '我搜了这本小说': 'I searched this novel.', '杨皎滢？': 'Yang Wei?', 'भारतीयों': 'Indians', '巴蜀': 'Bayu', '\x02tñ\x7f': '? tñ?', 'αβtαβ': 'aba', 'जल': 'water', 'बाध्य': 'Bound', 'মহাবিশ্ব': 'Universe', 'প্রসারিত': 'Stretch', 'अन्जाम': 'Anjaam', 'जीतना': 'win', 'कड़ा': 'Hard', '刁民': 'Untouchable', 'ขอให้พระเจ้าอยู่ด้วย': 'May God live too.', '油腻': 'Greasy', 'ᗯoᗰeᑎ': 'ᗯoᗰe ᑎ', 'להתראות': 'Goodbye', 'वाले': 'Ones', 'አየሁ': 'I saw', 'ओर': 'And', 'ずand': 'Without', 'निगोड़ा': 'Nigoda', 'эй': 'Hey'}

def pre_clean_unknown_words(text):
    for rare_word in unknown_to_english_map:
        if rare_word in text:
            text = text.replace(rare_word, unknown_to_english_map[rare_word])

    return text

rare_words_mapping = {' s.p ': ' ', ' S.P ': ' ', 'U.s.p': '', 'U.S.A.': 'USA', 'u.s.a.': 'USA', 'U.S.A': 'USA',
                      'u.s.a': 'USA', 'U.S.': 'USA', 'u.s.': 'USA', ' U.S ': ' USA ', ' u.s ': ' USA ', 'U.s.': 'USA',
                      ' U.s ': 'USA', ' u.S ': ' USA ', 'fu.k': 'fuck', 'U.K.': 'UK', ' u.k ': ' UK ',
                      ' don t ': ' do not ', 'bacteries': 'batteries', ' yr old ': ' years old ', 'Ph.D': 'PhD',
                      'cau.sing': 'causing', 'Kim Jong-Un': 'The president of North Korea', 'savegely': 'savagely',
                      'Ra apist': 'Rapist', '2fifth': 'twenty fifth', '2third': 'twenty third',
                      '2nineth': 'twenty nineth', '2fourth': 'twenty fourth', '#metoo': 'MeToo',
                      'Trumpcare': 'Trump health care system', '4fifth': 'forty fifth', 'Remainers': 'remainder',
                      'Terroristan': 'terrorist', 'antibrahmin': 'anti brahmin',
                      'fuckboys': 'fuckboy', 'Fuckboys': 'fuckboy', 'Fuckboy': 'fuckboy', 'fuckgirls': 'fuck girls',
                      'fuckgirl': 'fuck girl', 'Trumpsters': 'Trump supporters', '4sixth': 'forty sixth',
                      'culturr': 'culture',
                      'weatern': 'western', '4fourth': 'forty fourth', 'emiratis': 'emirates', 'trumpers': 'Trumpster',
                      'indans': 'indians', 'mastuburate': 'masturbate', 'f**k': 'fuck', 'F**k': 'fuck', 'F**K': 'fuck',
                      ' u r ': ' you are ', ' u ': ' you ', '操你妈': 'fuck your mother', 'e.g.': 'for example',
                      'i.e.': 'in other words', '...': '.', 'et.al': 'elsewhere', 'anti-Semitic': 'anti-semitic',
                      'f***': 'fuck', 'f**': 'fuc', 'F***': 'fuck', 'F**': 'fuc',
                      'a****': 'assho', 'a**': 'ass', 'h***': 'hole', 'A****': 'assho', 'A**': 'ass', 'H***': 'hole',
                      's***': 'shit', 's**': 'shi', 'S***': 'shit', 'S**': 'shi', 'Sh**': 'shit',
                      'p****': 'pussy', 'p*ssy': 'pussy', 'P****': 'pussy',
                      'p***': 'porn', 'p*rn': 'porn', 'P***': 'porn',
                      'st*up*id': 'stupid',
                      'd***': 'dick', 'di**': 'dick', 'h*ck': 'hack',
                      'b*tch': 'bitch', 'bi*ch': 'bitch', 'bit*h': 'bitch', 'bitc*': 'bitch', 'b****': 'bitch',
                      'b***': 'bitc', 'b**': 'bit', 'b*ll': 'bull'
                      }


def pre_clean_rare_words(text):
    for rare_word in rare_words_mapping:
        if rare_word in text:
            text = text.replace(rare_word, rare_words_mapping[rare_word])
    return text

# de-contract the contraction
def decontracted(text):
    # specific
    text = re.sub(r"(W|w)on(\'|\’)t ", "will not ", text)
    text = re.sub(r"(C|c)an(\'|\’)t ", "can not ", text)
    text = re.sub(r"(Y|y)(\'|\’)all ", "you all ", text)
    text = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", text)

    # general
    text = re.sub(r"(I|i)(\'|\’)m ", "i am ", text)
    text = re.sub(r"(A|a)in(\'|\’)t ", "is not ", text)
    text = re.sub(r"n(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)re ", " are ", text)
    text = re.sub(r"(\'|\’)s ", " is ", text)
    text = re.sub(r"(\'|\’)d ", " would ", text)
    text = re.sub(r"(\'|\’)ll ", " will ", text)
    text = re.sub(r"(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)ve ", " have ", text)
    return text

def clean_latex(text):
    """
    convert r"[math]\vec{x} + \vec{y}" to English
    """
    # edge case
    text = re.sub(r'\[math\]', ' LaTex math ', text)
    text = re.sub(r'\[\/math\]', ' LaTex math ', text)
    text = re.sub(r'\\', ' LaTex ', text)

    pattern_to_sub = {
        r'\\mathrm': ' LaTex math mode ',
        r'\\mathbb': ' LaTex math mode ',
        r'\\boxed': ' LaTex equation ',
        r'\\begin': ' LaTex equation ',
        r'\\end': ' LaTex equation ',
        r'\\left': ' LaTex equation ',
        r'\\right': ' LaTex equation ',
        r'\\(over|under)brace': ' LaTex equation ',
        r'\\text': ' LaTex equation ',
        r'\\vec': ' vector ',
        r'\\var': ' variable ',
        r'\\theta': ' theta ',
        r'\\mu': ' average ',
        r'\\min': ' minimum ',
        r'\\max': ' maximum ',
        r'\\sum': ' + ',
        r'\\times': ' * ',
        r'\\cdot': ' * ',
        r'\\hat': ' ^ ',
        r'\\frac': ' / ',
        r'\\div': ' / ',
        r'\\sin': ' Sine ',
        r'\\cos': ' Cosine ',
        r'\\tan': ' Tangent ',
        r'\\infty': ' infinity ',
        r'\\int': ' integer ',
        r'\\in': ' in ',
    }
    # post process for look up
    pattern_dict = {k.strip('\\'): v for k, v in pattern_to_sub.items()}
    # init re
    patterns = pattern_to_sub.keys()
    pattern_re = re.compile('(%s)' % '|'.join(patterns))

    def _replace(match):
        """
        reference: https://www.kaggle.com/hengzheng/attention-capsule-why-not-both-lb-0-694 # noqa
        """
        try:
            word = pattern_dict.get(match.group(0).strip('\\'))
        except KeyError:
            word = match.group(0)
            print('!!Error: Could Not Find Key: {}'.format(word))
        return word
    return pattern_re.sub(_replace, text)

# clean misspelling words
misspell_mapping = {'Terroristan': 'terrorist Pakistan', 'terroristan': 'terrorist Pakistan',
                    'FATF': 'Western summit conference',
                    'BIMARU': 'BIMARU Bihar, Madhya Pradesh, Rajasthan, Uttar Pradesh', 'Hinduphobic': 'Hindu phobic',
                    'hinduphobic': 'Hindu phobic', 'Hinduphobia': 'Hindu phobic', 'hinduphobia': 'Hindu phobic',
                    'Babchenko': 'Arkady Arkadyevich Babchenko faked death', 'Boshniaks': 'Bosniaks',
                    'Dravidanadu': 'Dravida Nadu', 'mysoginists': 'misogynists', 'MGTOWS': 'Men Going Their Own Way',
                    'mongloid': 'Mongoloid', 'unsincere': 'insincere', 'meninism': 'male feminism',
                    'jewplicate': 'jewish replicate', 'jewplicates': 'jewish replicate', 'andhbhakts': 'and Bhakt',
                    'unoin': 'Union', 'daesh': 'Islamic State of Iraq and the Levant', 'burnol': 'movement about Modi',
                    'Kalergi': 'Coudenhove-Kalergi', 'Bhakts': 'Bhakt', 'bhakts': 'Bhakt', 'Tambrahms': 'Tamil Brahmin',
                    'Pahul': 'Amrit Sanskar', 'SJW': 'social justice warrior', 'SJWs': 'social justice warrior',
                    ' incel': ' involuntary celibates', ' incels': ' involuntary celibates', 'emiratis': 'Emiratis',
                    'weatern': 'western', 'westernise': 'westernize', 'Pizzagate': 'debunked conspiracy theory',
                    'naïve': 'naive', 'Skripal': 'Russian military officer', 'Skripals': 'Russian military officer',
                    'Remainers': 'British remainer', 'Novichok': 'Soviet Union agents',
                    'gauri lankesh': 'Famous Indian Journalist', 'Castroists': 'Castro supporters',
                    'remainers': 'British remainer', 'bremainer': 'British remainer', 'antibrahmin': 'anti Brahminism',
                    'HYPSM': ' Harvard, Yale, Princeton, Stanford, MIT', 'HYPS': ' Harvard, Yale, Princeton, Stanford',
                    'kompromat': 'compromising material', 'Tharki': 'pervert', 'tharki': 'pervert',
                    'mastuburate': 'masturbate', 'Zoë': 'Zoe', 'indans': 'Indian', ' xender': ' gender',
                    'Naxali ': 'Naxalite ', 'Naxalities': 'Naxalites', 'Bathla': 'Namit Bathla',
                    'Mewani': 'Indian politician Jignesh Mevani', 'Wjy': 'Why',
                    'Fadnavis': 'Indian politician Devendra Fadnavis', 'Awadesh': 'Indian engineer Awdhesh Singh',
                    'Awdhesh': 'Indian engineer Awdhesh Singh', 'Khalistanis': 'Sikh separatist movement',
                    'madheshi': 'Madheshi', 'BNBR': 'Be Nice, Be Respectful',
                    'Jair Bolsonaro': 'Brazilian President politician', 'XXXTentacion': 'Tentacion',
                    'Slavoj Zizek': 'Slovenian philosopher',
                    'borderliners': 'borderlines', 'Brexit': 'British Exit', 'Brexiter': 'British Exit supporter',
                    'Brexiters': 'British Exit supporters', 'Brexiteer': 'British Exit supporter',
                    'Brexiteers': 'British Exit supporters', 'Brexiting': 'British Exit',
                    'Brexitosis': 'British Exit disorder', 'brexit': 'British Exit',
                    'brexiters': 'British Exit supporters', 'jallikattu': 'Jallikattu', 'fortnite': 'Fortnite',
                    'Swachh': 'Swachh Bharat mission campaign ', 'Quorans': 'Quora users', 'Qoura': 'Quora',
                    'quoras': 'Quora', 'Quroa': 'Quora', 'QUORA': 'Quora', 'Stupead': 'stupid',
                    'narcissit': 'narcissist', 'trigger nometry': 'trigonometry',
                    'trigglypuff': 'student Criticism of Conservatives', 'peoplelook': 'people look',
                    'paedophelia': 'paedophilia', 'Uogi': 'Yogi', 'adityanath': 'Adityanath',
                    'Yogi Adityanath': 'Indian monk and Hindu nationalist politician',
                    'Awdhesh Singh': 'Commissioner of India', 'Doklam': 'Tibet', 'Drumpf ': 'Donald Trump fool ',
                    'Drumpfs': 'Donald Trump fools', 'Strzok': 'Hillary Clinton scandal', 'rohingya': 'Rohingya ',
                    ' wumao ': ' cheap Chinese stuff ', 'wumaos': 'cheap Chinese stuff', 'Sanghis': 'Sanghi',
                    'Tamilans': 'Tamils', 'biharis': 'Biharis', 'Rejuvalex': 'hair growth formula Medicine',
                    'Fekuchand': 'PM Narendra Modi in India', 'feku': 'Feku', 'Chaiwala': 'tea seller in India',
                    'Feku': 'PM Narendra Modi in India ', 'deplorables': 'deplorable', 'muhajirs': 'Muslim immigrant',
                    'Gujratis': 'Gujarati', 'Chutiya': 'Tibet people ', 'Chutiyas': 'Tibet people ',
                    'thighing': 'masterbate between the legs of a female infant', '卐': 'Nazi Germany',
                    'Pribumi': 'Native Indonesian', 'Gurmehar': 'Gurmehar Kaur Indian student activist',
                    'Khazari': 'Khazars', 'Demonetization': 'demonetization', 'demonetisation': 'demonetization',
                    'demonitisation': 'demonetization', 'demonitization': 'demonetization',
                    'antinationals': 'antinational', 'Cryptocurrencies': 'cryptocurrency',
                    'cryptocurrencies': 'cryptocurrency', 'Hindians': 'North Indian', 'Hindian': 'North Indian',
                    'vaxxer': 'vocal nationalist ', 'remoaner': 'remainer ', 'bremoaner': 'British remainer ',
                    'Jewism': 'Judaism', 'Eroupian': 'European', "J & K Dy CM H ' ble Kavinderji": '',
                    'WMAF': 'White male married Asian female', 'AMWF': 'Asian male married White female',
                    'moeslim': 'Muslim', 'cishet': 'cisgender and heterosexual person', 'Eurocentrics': 'Eurocentrism',
                    'Eurocentric': 'Eurocentrism', 'Afrocentrics': 'Africa centrism', 'Afrocentric': 'Africa centrism',
                    'Jewdar': 'Jew dar', 'marathis': 'Marathi', 'Gynophobic': 'Gyno phobic',
                    'Trumpanzees': 'Trump chimpanzee fool', 'Crimean': 'Crimea people ', 'atrracted': 'attract',
                    'Myeshia': 'widow of Green Beret killed in Niger', 'demcoratic': 'Democratic', 'raaping': 'raping',
                    'feminazism': 'feminism nazi', 'langague': 'language', 'sathyaraj': 'actor',
                    'Hongkongese': 'HongKong people', 'hongkongese': 'HongKong people', 'Kashmirians': 'Kashmirian',
                    'Chodu': 'fucker', 'penish': 'penis',
                    'chitpavan konkanastha': 'Hindu Maharashtrian Brahmin community',
                    'Madridiots': 'Real Madrid idiot supporters', 'Ambedkarite': 'Dalit Buddhist movement ',
                    'ReleaseTheMemo': 'cry for the right and Trump supporters', 'harrase': 'harass',
                    'Barracoon': 'Black slave', 'Castrater': 'castration', 'castrater': 'castration',
                    'Rapistan': 'Pakistan rapist', 'rapistan': 'Pakistan rapist', 'Turkified': 'Turkification',
                    'turkified': 'Turkification', 'Dumbassistan': 'dumb ass Pakistan', 'facetards': 'Facebook retards',
                    'rapefugees': 'rapist refugee', 'Khortha': 'language in the Indian state of Jharkhand',
                    'Magahi': 'language in the northeastern Indian', 'Bajjika': 'language spoken in eastern India',
                    'superficious': 'superficial', 'Sense8': 'American science fiction drama web television series',
                    'Saipul Jamil': 'Indonesia artist', 'bhakht': 'bhakti', 'Smartia': 'dumb nation',
                    'absorve': 'absolve', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Whta': 'What',
                    'esspecial': 'especial', 'doI': 'do I', 'theBest': 'the best',
                    'howdoes': 'how does', 'Etherium': 'Ethereum', '2k17': '2017', '2k18': '2018', 'qiblas': 'qibla',
                    'Hello4 2 cab': 'Online Cab Booking', 'bodyshame': 'body shaming', 'bodyshoppers': 'body shopping',
                    'bodycams': 'body cams', 'Cananybody': 'Can any body', 'deadbody': 'dead body',
                    'deaddict': 'de addict', 'Northindian': 'North Indian ', 'northindian': 'north Indian ',
                    'northkorea': 'North Korea', 'koreaboo': 'Korea boo ',
                    'Brexshit': 'British Exit bullshit', 'shitpost': 'shit post', 'shitslam': 'shit Islam',
                    'shitlords': 'shit lords', 'Fck': 'Fuck', 'Clickbait': 'click bait ', 'clickbait': 'click bait ',
                    'mailbait': 'mail bait', 'healhtcare': 'healthcare', 'trollbots': 'troll bots',
                    'trollled': 'trolled', 'trollimg': 'trolling', 'cybertrolling': 'cyber trolling',
                    'sickular': 'India sick secular ', 'Idiotism': 'idiotism',
                    'Niggerism': 'Nigger', 'Niggeriah': 'Nigger'}

def clean_misspell(text):
    for bad_word in misspell_mapping:
        if bad_word in text:
            text = text.replace(bad_word, misspell_mapping[bad_word])
    return text

regular_punct = list(string.punctuation)
extra_punct = [
    ',', '.', '"', ':', ')', '(', '!', '?', '|', ';', "'", '$', '&',
    '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',
    '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',
    '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
    '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',
    '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
    'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
    '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
    '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤']
all_punct = list(set(regular_punct + extra_punct))
# do not spacing - and .
all_punct.remove('-')
all_punct.remove('.')

def spacing_punctuation(text):
    """
    add space before and after punctuation and symbols
    """
    for punc in all_punct:
        if punc in text:
            text = text.replace(punc, f' {punc} ')
    return text

# spell check and according to bad case analyse
bad_case_words = {'jewprofits': 'jew profits', 'QMAS': 'Quality Migrant Admission Scheme', 'casterating': 'castrating',
                  'Kashmiristan': 'Kashmir', 'CareOnGo': 'India first and largest Online distributor of medicines',
                  'Setya Novanto': 'a former Indonesian politician', 'TestoUltra': 'male sexual enhancement supplement',
                  'rammayana': 'ramayana', 'Badaganadu': 'Brahmin community that mainly reside in Karnataka',
                  'bitcjes': 'bitches', 'mastubrate': 'masturbate', 'Français': 'France',
                  'Adsresses': 'address', 'flemmings': 'flemming', 'intermate': 'inter mating', 'feminisam': 'feminism',
                  'cuckholdry': 'cuckold', 'Niggor': 'black hip-hop and electronic artist', 'narcsissist': 'narcissist',
                  'Genderfluid': 'Gender fluid', ' Im ': ' I am ', ' dont ': ' do not ', 'Qoura': 'Quora',
                  'ethethnicitesnicites': 'ethnicity', 'Namit Bathla': 'Content Writer', 'What sApp': 'WhatsApp',
                  'Führer': 'Fuhrer', 'covfefe': 'coverage', 'accedentitly': 'accidentally', 'Cuckerberg': 'Zuckerberg',
                  'transtrenders': 'incredibly disrespectful to real transgender people',
                  'frozen tamod': 'Pornographic website', 'hindians': 'North Indian', 'hindian': 'North Indian',
                  'celibatess': 'celibates', 'Trimp': 'Trump', 'wanket': 'wanker', 'wouldd': 'would',
                  'arragent': 'arrogant', 'Ra - apist': 'rapist', 'idoot': 'idiot', 'gangstalkers': 'gangs talkers',
                  'toastsexual': 'toast sexual', 'inapropriately': 'inappropriately', 'dumbassess': 'dumbass',
                  'germanized': 'become german', 'helisexual': 'sexual', 'regilious': 'religious',
                  'timetraveller': 'time traveller', 'darkwebcrawler': 'dark webcrawler', 'routez': 'route',
                  'trumpians': 'Trump supporters', 'irreputable': 'reputation', 'serieusly': 'seriously',
                  'anti cipation': 'anticipation', 'microaggression': 'micro aggression', 'Afircans': 'Africans',
                  'microapologize': 'micro apologize', 'Vishnus': 'Vishnu', 'excritment': 'excitement',
                  'disagreemen': 'disagreement', 'gujratis': 'gujarati', 'gujaratis': 'gujarati',
                  'ugggggggllly': 'ugly',
                  'Germanity': 'German', 'SoyBoys': 'cuck men lacking masculine characteristics',
                  'н': 'h', 'м': 'm', 'ѕ': 's', 'т': 't', 'в': 'b', 'υ': 'u', 'ι': 'i',
                  'genetilia': 'genitalia', 'r - apist': 'rapist', 'Borokabama': 'Barack Obama',
                  'arectifier': 'rectifier', 'pettypotus': 'petty potus', 'magibabble': 'magi babble',
                  'nothinking': 'thinking', 'centimiters': 'centimeters', 'saffronized': 'India, politics, derogatory',
                  'saffronize': 'India, politics, derogatory', ' incect ': ' insect ', 'weenus': 'elbow skin',
                  'Pakistainies': 'Pakistanis', 'goodspeaks': 'good speaks', 'inpregnated': 'in pregnant',
                  'rapefilms': 'rape films', 'rapiest': 'rapist', 'hatrednesss': 'hatred',
                  'heightism': 'height discrimination', 'getmy': 'get my', 'onsocial': 'on social',
                  'worstplatform': 'worst platform', 'platfrom': 'platform', 'instagate': 'instigate',
                  'Loy Machedeo': 'person', ' dsire ': ' desire ', 'iservant': 'servant', 'intelliegent': 'intelligent',
                  'WW 1': ' WW1 ', 'WW 2': ' WW2 ', 'ww 1': ' WW1 ', 'ww 2': ' WW2 ',
                  'keralapeoples': 'kerala peoples', 'trumpervotes': 'trumper votes', 'fucktrumpet': 'fuck trumpet',
                  'likebJaish': 'like bJaish', 'likemy': 'like my', 'Howlikely': 'How likely',
                  'disagreementts': 'disagreements', 'disagreementt': 'disagreement',
                  'meninist': "male chauvinism", 'feminists': 'feminism supporters', 'Ghumendra': 'Bhupendra',
                  'emellishments': 'embellishments',
                  'settelemen': 'settlement',
                  'Richmencupid': 'rich men dating website', 'richmencupid': 'rich men dating website',
                  'Gaudry - Schost': '', 'ladymen': 'ladyboy', 'hasserment': 'Harassment',
                  'instrumentalizing': 'instrument', 'darskin': 'dark skin', 'balckwemen': 'balck women',
                  'recommendor': 'recommender', 'wowmen': 'women', 'expertthink': 'expert think',
                  'whitesplaining': 'white splaining', 'Inquoraing': 'inquiring', 'whilemany': 'while many',
                  'manyother': 'many other', 'involvedinthe': 'involved in the', 'slavetrade': 'slave trade',
                  'aswell': 'as well', 'fewshowanyRemorse': 'few show any Remorse', 'trageting': 'targeting',
                  'getile': 'gentile', 'Gujjus': 'derogatory Gujarati', 'judisciously': 'judiciously',
                  'Hue Mungus': 'feminist bait', 'Hugh Mungus': 'feminist bait', 'Hindustanis': '',
                  'Virushka': 'Great Relationships Couple', 'exclusinary': 'exclusionary', 'himdus': 'hindus',
                  'Milo Yianopolous': 'a British polemicist', 'hidusim': 'hinduism',
                  'holocaustable': 'holocaust', 'evangilitacal': 'evangelical', 'Busscas': 'Buscas',
                  'holocaustal': 'holocaust', 'incestious': 'incestuous', 'Tennesseus': 'Tennessee',
                  'GusDur': 'Gus Dur',
                  'RPatah - Tan Eng Hwan': 'Silsilah', 'Reinfectus': 'reinfect', 'pharisaistic': 'pharisaism',
                  'nuslims': 'Muslims', 'taskus': '', 'musims': 'Muslims',
                  'Musevi': 'the independence of Mexico', ' racious ': 'discrimination expression of racism',
                  'Muslimophobia': 'Muslim phobia', 'justyfied': 'justified', 'holocause': 'holocaust',
                  'musilim': 'Muslim', 'misandrous': 'misandry', 'glrous': 'glorious', 'desemated': 'decimated',
                  'votebanks': 'vote banks', 'Parkistan': 'Pakistan', 'Eurooe': 'Europe', 'animlaistic': 'animalistic',
                  'Asiasoid': 'Asian', 'Congoid': 'Congolese', 'inheritantly': 'inherently',
                  'Asianisation': 'Becoming Asia',
                  'Russosphere': 'russia sphere of influence', 'exMuslims': 'Ex-Muslims',
                  'discriminatein': 'discrimination', ' hinus ': ' hindus ', 'Nibirus': 'Nibiru',
                  'habius - corpus': 'habeas corpus', 'prentious': 'pretentious', 'Sussia': 'ancient Jewish village',
                  'moustachess': 'moustaches', 'Russions': 'Russians', 'Yuguslavia': 'Yugoslavia',
                  'atrocitties': 'atrocities', 'Muslimophobe': 'Muslim phobic', 'fallicious': 'fallacious',
                  'recussed': 'recursed', '@ usafmonitor': '', 'lustfly': 'lustful', 'canMuslims': 'can Muslims',
                  'journalust': 'journalist', 'digustingly': 'disgustingly', 'harasing': 'harassing',
                  'greatuncle': 'great uncle', 'Drumpf': 'Trump', 'rejectes': 'rejected', 'polyagamous': 'polygamous',
                  'Mushlims': 'Muslims', 'accusition': 'accusation', 'geniusses': 'geniuses',
                  'moustachesomething': 'moustache something', 'heineous': 'heinous',
                  'Sapiosexuals': 'sapiosexual', 'sapiosexuals': 'sapiosexual', 'Sapiosexual': 'sapiosexual',
                  'sapiosexual': 'Sexually attracted to intelligence', 'pansexuals': 'pansexual',
                  'autosexual': 'auto sexual', 'sexualSlutty': 'sexual Slutty', 'hetorosexuality': 'hetoro sexuality',
                  'chinesese': 'chinese', 'pizza gate': 'debunked conspiracy theory',
                  'countryless': 'Having no country',
                  'muslimare': 'Muslim are', 'iPhoneX': 'iPhone', 'lionese': 'lioness', 'marionettist': 'Marionettes',
                  'demonetize': 'demonetized', 'eneyone': 'anyone', 'Karonese': 'Karo people Indonesia',
                  'minderheid': 'minder worse', 'mainstreamly': 'mainstream', 'contraproductive': 'contra productive',
                  'diffenky': 'differently', 'abandined': 'abandoned', 'p0 rnstars': 'pornstars',
                  'overproud': 'over proud',
                  'cheekboned': 'cheek boned', 'heriones': 'heroines', 'eventhogh': 'even though',
                  'americanmedicalassoc': 'american medical assoc', 'feelwhen': 'feel when', 'Hhhow': 'how',
                  'reallySemites': 'really Semites', 'gamergaye': 'gamersgate', 'manspreading': 'man spreading',
                  'thammana': 'Tamannaah Bhatia', 'dogmans': 'dogmas', 'managementskills': 'management skills',
                  'mangoliod': 'mongoloid', 'geerymandered': 'gerrymandered', 'mandateing': 'man dateing',
                  'Romanium': 'Romanum',
                  'mailwoman': 'mail woman', 'humancoalition': 'human coalition',
                  'manipullate': 'manipulate', 'everyo0 ne': 'everyone', 'takeove': 'takeover',
                  'Nonchristians': 'Non Christians', 'goverenments': 'governments', 'govrment': 'government',
                  'polygomists': 'polygamists', 'Demogorgan': 'Demogorgon', 'maralago': 'Mar-a-Lago',
                  'antibigots': 'anti bigots', 'gouing': 'going', 'muzaffarbad': 'muzaffarabad',
                  'suchvstupid': 'such stupid', 'apartheidisrael': 'apartheid israel', 
                  'personaltiles': 'personal titles', 'lawyergirlfriend': 'lawyer girl friend',
                  'northestern': 'northwestern', 'yeardold': 'years old', 'masskiller': 'mass killer',
                  'southeners': 'southerners', 'Unitedstatesian': 'United states',

                  'peoplekind': 'people kind', 'peoplelike': 'people like', 'countrypeople': 'country people',
                  'shitpeople': 'shit people', 'trumpology': 'trump ology', 'trumpites': 'Trump supporters',
                  'trumplies': 'trump lies', 'donaldtrumping': 'donald trumping', 'trumpdating': 'trump dating',
                  'trumpsters': 'trumpeters', 'ciswomen': 'cis women', 'womenizer': 'womanizer',
                  'pregnantwomen': 'pregnant women', 'autoliker': 'auto liker', 'smelllike': 'smell like',
                  'autolikers': 'auto likers', 'religiouslike': 'religious like', 'likemail': 'like mail',
                  'fislike': 'dislike', 'sneakerlike': 'sneaker like', 'like⬇': 'like',
                  'likelovequotes': 'like lovequotes', 'likelogo': 'like logo', 'sexlike': 'sex like',
                  'Whatwould': 'What would', 'Howwould': 'How would', 'manwould': 'man would',
                  'exservicemen': 'ex servicemen', 'femenism': 'feminism', 'devopment': 'development',
                  'doccuments': 'documents', 'supplementplatform': 'supplement platform', 'mendatory': 'mandatory',
                  'moviments': 'movements', 'Kremenchuh': 'Kremenchug', 'docuements': 'documents',
                  'determenism': 'determinism', 'envisionment': 'envision ment',
                  'tricompartmental': 'tri compartmental', 'AddMovement': 'Add Movement',
                  'mentionong': 'mentioning', 'Whichtreatment': 'Which treatment', 'repyament': 'repayment',
                  'insemenated': 'inseminated', 'inverstment': 'investment',
                  'managemental': 'manage mental', 'Inviromental': 'Environmental', 'menstrution': 'menstruation',
                  'indtrument': 'instrument', 'mentenance': 'maintenance', 'fermentqtion': 'fermentation',
                  'achivenment': 'achievement', 'mismanagements': 'mis managements', 'requriment': 'requirement',
                  'denomenator': 'denominator', 'drparment': 'department', 'acumens': 'acumen s',
                  'celemente': 'Clemente', 'manajement': 'management', 'govermenent': 'government',
                  'accomplishmments': 'accomplishments', 'rendementry': 'rendement ry',
                  'repariments': 'departments', 'menstrute': 'menstruate', 'determenistic': 'deterministic',
                  'resigment': 'resignment', 'selfpayment': 'self payment', 'imrpovement': 'improvement',
                  'enivironment': 'environment', 'compartmentley': 'compartment',
                  'augumented': 'augmented', 'parmenent': 'permanent', 'dealignment': 'de alignment',
                  'develepoments': 'developments', 'menstrated': 'menstruated', 'phnomenon': 'phenomenon',
                  'Employmment': 'Employment', 'dimensionalise': 'dimensional ise', 'menigioma': 'meningioma',
                  'recrument': 'recrement', 'Promenient': 'Provenient', 'gonverment': 'government',
                  'statemment': 'statement', 'recuirement': 'requirement', 'invetsment': 'investment',
                  'parilment': 'parchment', 'parmently': 'patiently', 'agreementindia': 'agreement india',
                  'menifesto': 'manifesto', 'accomplsihments': 'accomplishments', 'disangagement': 'disengagement',
                  'aevelopment': 'development', 'procument': 'procumbent', 'harashment': 'harassment',
                  'Tiannanmen': 'Tiananmen', 'commensalisms': 'commensal isms', 'devlelpment': 'development',
                  'dimensons': 'dimensions', 'recruitment2017': 'recruitment 2017', 'polishment': 'pol ishment',
                  'CommentSafe': 'Comment Safe', 'meausrements': 'measurements', 'geomentrical': 'geometrical',
                  'undervelopment': 'undevelopment', 'mensurational': 'mensuration al', 'fanmenow': 'fan menow',
                  'permenganate': 'permanganate', 'bussinessmen': 'businessmen',
                  'supertournaments': 'super tournaments', 'permanmently': 'permanently',
                  'lamenectomy': 'lamnectomy', 'assignmentcanyon': 'assignment canyon', 'adgestment': 'adjustment',
                  'mentalized': 'metalized', 'docyments': 'documents', 'requairment': 'requirement',
                  'batsmencould': 'batsmen could', 'argumentetc': 'argument etc', 'enjoiment': 'enjoyment',
                  'invement': 'movement', 'accompliushments': 'accomplishments', 'regements': 'regiments',
                  'departmentHow': 'department How', 'Aremenian': 'Armenian', 'amenclinics': 'amen clinics',
                  'nonfermented': 'non fermented', 'Instumentation': 'Instrumentation', 'mentalitiy': 'mentality',
                  ' govermen ': 'goverment', 'underdevelopement': 'under developement', 'parlimentry': 'parliamentary',
                  'indemenity': 'indemnity', 'Inatrumentation': 'Instrumentation', 'menedatory': 'mandatory',
                  'mentiri': 'entire', 'accomploshments': 'accomplishments', 'instrumention': 'instrument ion',
                  'afvertisements': 'advertisements', 'parlementarian': 'parlement arian',
                  'entitlments': 'entitlements', 'endrosment': 'endorsement', 'improment': 'impriment',
                  'archaemenid': 'Achaemenid', 'replecement': 'replacement', 'placdment': 'placement',
                  'femenise': 'feminise', 'envinment': 'environment', 'AmenityCompany': 'Amenity Company',
                  'increaments': 'increments', 'accomplihsments': 'accomplishments',
                  'manygovernment': 'many government', 'panishments': 'punishments', 'elinment': 'eloinment',
                  'mendalin': 'mend alin', 'farmention': 'farm ention', 'preincrement': 'pre increment',
                  'postincrement': 'post increment', 'achviements': 'achievements', 'menditory': 'mandatory',
                  'Emouluments': 'Emoluments', 'Stonemen': 'Stone men', 'menmium': 'medium',
                  'entaglement': 'entanglement', 'integumen': 'integument', 'harassument': 'harassment',
                  'retairment': 'retainment', 'enviorement': 'environment', 'tormentous': 'torment ous',
                  'confiment': 'confident', 'Enchroachment': 'Encroachment', 'prelimenary': 'preliminary',
                  'fudamental': 'fundamental', 'instrumenot': 'instrument', 'icrement': 'increment',
                  'prodimently': 'prominently', 'meniss': 'menise', 'Whoimplemented': 'Who implemented',
                  'Representment': 'Rep resentment', 'StartFragment': 'Start Fragment',
                  'EndFragment': 'End Fragment', ' documentarie ': ' documentaries ', 'requriments': 'requirements',
                  'constitutionaldevelopment': 'constitutional development', 'parlamentarians': 'parliamentarians',
                  'Rumenova': 'Rumen ova', 'argruments': 'arguments', 'findamental': 'fundamental',
                  'totalinvestment': 'total investment', 'gevernment': 'government', 'recmommend': 'recommend',
                  'appsmoment': 'apps moment', 'menstruual': 'menstrual', 'immplemented': 'implemented',
                  'engangement': 'engagement', 'invovement': 'involvement', 'returement': 'retirement',
                  'simentaneously': 'simultaneously', 'accompishments': 'accomplishments',
                  'menstraution': 'menstruation', 'experimently': 'experiment', 'abdimen': 'abdomen',
                  'cemenet': 'cement', 'propelment': 'propel ment', 'unamendable': 'un amendable',
                  'employmentnews': 'employment news', 'lawforcement': 'law forcement',
                  'menstuating': 'menstruating', 'fevelopment': 'development', 'reglamented': 'reg lamented',
                  'imrovment': 'improvement', 'recommening': 'recommending', 'sppliment': 'supplement',
                  'measument': 'measurement', 'reimbrusement': 'reimbursement', 'Nutrament': 'Nutriment',
                  'puniahment': 'punishment', 'subligamentous': 'sub ligamentous', 'comlementry': 'complementary',
                  'reteirement': 'retirement', 'envioronments': 'environments', 'haraasment': 'harassment',
                  'USAgovernment': 'USA government', 'Apartmentfinder': 'Apartment finder',
                  'encironment': 'environment', 'metacompartment': 'meta compartment',
                  'augumentation': 'argumentation', 'dsymenorrhoea': 'dysmenorrhoea',
                  'nonabandonment': 'non abandonment', 'annoincement': 'announcement',
                  'menberships': 'memberships', 'Gamenights': 'Game nights', 'enliightenment': 'enlightenment',
                  'supplymentry': 'supplementary', 'parlamentary': 'parliamentary', 'duramen': 'dura men',
                  'hotelmanagement': 'hotel management', 'deartment': 'department',
                  'treatmentshelp': 'treatments help', 'attirements': 'attire ments',
                  'amendmending': 'amend mending', 'pseudomeningocele': 'pseudo meningocele',
                  'intrasegmental': 'intra segmental', 'treatmenent': 'treatment', 'infridgement': 'infringement',
                  'infringiment': 'infringement', 'recrecommend': 'rec recommend', 'entartaiment': 'entertainment',
                  'inplementing': 'implementing', 'indemendent': 'independent', 'tremendeous': 'tremendous',
                  'commencial': 'commercial', 'scomplishments': 'accomplishments', 'Emplement': 'Implement',
                  'dimensiondimensions': 'dimension dimensions', 'depolyment': 'deployment',
                  'conpartment': 'compartment', 'govnments': 'movements', 'menstrat': 'menstruate',
                  'accompplishments': 'accomplishments', 'Enchacement': 'Enchancement',
                  'developmenent': 'development', 'emmenagogues': 'emmenagogue', 'aggeement': 'agreement',
                  'elementsbond': 'elements bond', 'remenant': 'remnant', 'Manamement': 'Management',
                  'Augumented': 'Augmented', 'dimensonless': 'dimensionless',
                  'ointmentsointments': 'ointments ointments', 'achiements': 'achievements',
                  'recurtment': 'recurrent', 'gouverments': 'governments', 'docoment': 'document',
                  'programmingassignments': 'programming assignments', 'menifest': 'manifest',
                  'investmentguru': 'investment guru', 'deployements': 'deployments', 'Invetsment': 'Investment',
                  'plaement': 'placement', 'Perliament': 'Parliament', 'femenists': 'feminists',
                  'ecumencial': 'ecumenical', 'advamcements': 'advancements', 'refundment': 'refund ment',
                  'settlementtake': 'settlement take', 'mensrooms': 'mens rooms',
                  'productManagement': 'product Management', 'armenains': 'armenians',
                  'betweenmanagement': 'between management', 'difigurement': 'disfigurement',
                  'Armenized': 'Armenize', 'hurrasement': 'hurra sement', 'mamgement': 'management',
                  'momuments': 'monuments', 'eauipments': 'equipments', 'managemenet': 'management',
                  'treetment': 'treatment', 'webdevelopement': 'web developement', 'supplemenary': 'supplementary',
                  'Encironmental': 'Environmental', 'Understandment': 'Understand ment',
                  'enrollnment': 'enrollment', 'thinkstrategic': 'think strategic', 'thinkinh': 'thinking',
                  'Softthinks': 'Soft thinks', 'underthinking': 'under thinking', 'thinksurvey': 'think survey',
                  'whitelash': 'white lash', 'whiteheds': 'whiteheads', 'whitetning': 'whitening',
                  'whitegirls': 'white girls', 'whitewalkers': 'white walkers', 'manycountries': 'many countries',
                  'accomany': 'accompany', 'fromGermany': 'from Germany', 'manychat': 'many chat',
                  'Germanyl': 'Germany l', 'manyness': 'many ness', 'many4': 'many', 'exmuslims': 'ex muslims',
                  'digitizeindia': 'digitize india', 'indiarush': 'india rush', 'indiareads': 'india reads',
                  'telegraphindia': 'telegraph india', 'Southindia': 'South india', 'Airindia': 'Air india',
                  'siliconindia': 'silicon india', 'airindia': 'air india', 'indianleaders': 'indian leaders',
                  'fundsindia': 'funds india', 'indianarmy': 'indian army', 'Technoindia': 'Techno india',
                  'Betterindia': 'Better india', 'capesindia': 'capes india', 'Rigetti': 'Ligetti',
                  'vegetablr': 'vegetable', 'get90': 'get', 'Magetta': 'Maretta', 'nagetive': 'native',
                  'isUnforgettable': 'is Unforgettable', 'get630': 'get 630', 'GadgetPack': 'Gadget Pack',
                  'Languagetool': 'Language tool', 'bugdget': 'budget', 'africaget': 'africa get',
                  'ABnegetive': 'Abnegative', 'orangetheory': 'orange theory', 'getsmuggled': 'get smuggled',
                  'avegeta': 'ave geta', 'gettubg': 'getting', 'gadgetsnow': 'gadgets now',
                  'surgetank': 'surge tank', 'gadagets': 'gadgets', 'getallparts': 'get allparts',
                  'messenget': 'messenger', 'vegetarean': 'vegetarian', 'get1000': 'get 1000',
                  'getfinancing': 'get financing', 'getdrip': 'get drip', 'AdsTargets': 'Ads Targets',
                  'tgethr': 'together', 'vegetaries': 'vegetables', 'forgetfulnes': 'forgetfulness',
                  'fisgeting': 'fidgeting', 'BudgetAir': 'Budget Air',
                  'getDepersonalization': 'get Depersonalization', 'negetively': 'negatively',
                  'gettibg': 'getting', 'nauget': 'naught', 'Bugetti': 'Bugatti', 'plagetum': 'plage tum',
                  'vegetabale': 'vegetable', 'changetip': 'change tip', 'blackwashing': 'black washing',
                  'blackpink': 'black pink', 'blackmoney': 'black money',
                  'blackmarks': 'black marks', 'blackbeauty': 'black beauty', 'unblacklisted': 'un blacklisted',
                  'blackdotes': 'black dotes', 'blackboxing': 'black boxing', 'blackpaper': 'black paper',
                  'blackpower': 'black power', 'Latinamericans': 'Latin americans', 'musigma': 'mu sigma',
                  'Indominus': 'In dominus', 'usict': 'USSCt', 'indominus': 'in dominus', 'Musigma': 'Mu sigma',
                  'plus5': 'plus', 'Russiagate': 'Russia gate', 'russophobic': 'Russophobiac',
                  'Marcusean': 'Marcuse an', 'Radijus': 'Radius', 'cobustion': 'combustion',
                  'Austrialians': 'Australians', 'mylogenous': 'myogenous', 'Raddus': 'Radius',
                  'hetrogenous': 'heterogenous', 'greenhouseeffect': 'greenhouse effect', 'aquous': 'aqueous',
                  'Taharrush': 'Tahar rush', 'Senousa': 'Venous', 'diplococcus': 'diplo coccus',
                  'CityAirbus': 'City Airbus', 'sponteneously': 'spontaneously', 'trustless': 't rustless',
                  'Pushkaram': 'Pushkara m', 'Fusanosuke': 'Fu sanosuke', 'isthmuses': 'isthmus es',
                  'lucideus': 'lucidum', 'overjustification': 'over justification', 'Bindusar': 'Bind usar',
                  'cousera': 'couler', 'musturbation': 'masturbation', 'infustry': 'industry',
                  'Huswifery': 'Huswife ry', 'rombous': 'bombous', 'disengenuously': 'disingenuously',
                  'sllybus': 'syllabus', 'celcious': 'delicious', 'cellsius': 'celsius',
                  'lethocerus': 'Lethocerus', 'monogmous': 'monogamous', 'Ballyrumpus': 'Bally rumpus',
                  'Koushika': 'Koushik a', 'vivipoarous': 'viviparous', 'ludiculous': 'ridiculous',
                  'sychronous': 'synchronous', 'industiry': 'industry', 'scuduse': 'scud use',
                  'babymust': 'baby must', 'simultqneously': 'simultaneously', 'exust': 'ex ust',
                  'notmusing': 'not musing', 'Zamusu': 'Amuse', 'tusaki': 'tu saki', 'Marrakush': 'Marrakesh',
                  'justcheaptickets': 'just cheaptickets', 'Ayahusca': 'Ayahausca', 'samousa': 'samosa',
                  'Gusenberg': 'Gutenberg', 'illustratuons': 'illustrations', 'extemporeneous': 'extemporaneous',
                  'Mathusla': 'Mathusala', 'Confundus': 'Con fundus', 'tusts': 'trusts', 'poisenious': 'poisonous',
                  'Mevius': 'Medius', 'inuslating': 'insulating', 'aroused21000': 'aroused 21000',
                  'Wenzeslaus': 'Wenceslaus', 'JustinKase': 'Justin Kase', 'purushottampur': 'purushottam pur',
                  'citruspay': 'citrus pay', 'secutus': 'sects', 'austentic': 'austenitic',
                  'FacePlusPlus': 'Face PlusPlus', 'aysnchronous': 'asynchronous',
                  'teamtreehouse': 'team treehouse', 'uncouncious': 'unconscious', 'Priebuss': 'Prie buss',
                  'consciousuness': 'consciousness', 'susubsoil': 'su subsoil', 'trimegistus': 'Trismegistus',
                  'protopeterous': 'protopterous', 'trustworhty': 'trustworthy', 'ushually': 'usually',
                  'industris': 'industries', 'instantneous': 'instantaneous', 'superplus': 'super plus',
                  'shrusti': 'shruti', 'hindhus': 'hindus', 'outonomous': 'autonomous', 'reliegious': 'religious',
                  'Kousakis': 'Kou sakis', 'reusult': 'result', 'JanusGraph': 'Janus Graph',
                  'palusami': 'palus ami', 'mussraff': 'muss raff', 'hukous': 'humous',
                  'photoacoustics': 'photo acoustics', 'kushanas': 'kusha nas', 'justdile': 'justice',
                  'Massahusetts': 'Massachusetts', 'uspset': 'upset', 'sustinet': 'sustinent',
                  'consicious': 'conscious', 'Sadhgurus': 'Sadh gurus', 'hystericus': 'hysteric us',
                  'visahouse': 'visa house', 'supersynchronous': 'super synchronous', 'posinous': 'rosinous',
                  'Fernbus': 'Fern bus', 'Tiltbrush': 'Tilt brush', 'glueteus': 'gluteus', 'posionus': 'poisons',
                  'Freus': 'Frees', 'Zhuchengtyrannus': 'Zhucheng tyrannus', 'savonious': 'sanious',
                  'CusJo': 'Cusco', 'congusion': 'confusion', 'dejavus': 'dejavu s', 'uncosious': 'uncopious',
                  'previius': 'previous', 'counciousness': 'conciousness', 'lustorus': 'lustrous',
                  'sllyabus': 'syllabus', 'mousquitoes': 'mosquitoes', 'Savvius': 'Savvies', 'arceius': 'Arcesius',
                  'prejusticed': 'prejudiced', 'requsitioned': 'requisitioned',
                  'deindustralization': 'deindustrialization', 'muscleblaze': 'muscle blaze',
                  'ConsciousX5': 'conscious', 'nitrogenious': 'nitrogenous', 'mauritious': 'mauritius',
                  'rigrously': 'rigorously', 'Yutyrannus': 'Yu tyrannus', 'muscualr': 'muscular',
                  'conscoiusness': 'consciousness', 'Causians': 'Crusians', 'WorkFusion': 'Work Fusion',
                  'puspak': 'pu spak', 'Inspirus': 'Inspires', 'illiustrations': 'illustrations',
                  'Nobushi': 'No bushi', 'theuseof': 'thereof', 'suspicius': 'suspicious', 'Intuous': 'Virtuous',
                  'gaushalas': 'gaus halas', 'campusthrough': 'campus through', 'seriousity': 'seriosity',
                  'resustence': 'resistence', 'geminatus': 'geminates', 'disquss': 'discuss',
                  'nicholus': 'nicholas', 'Husnai': 'Hussar', 'diiscuss': 'discuss', 'diffussion': 'diffusion',
                  'phusicist': 'physicist', 'ernomous': 'enormous', 'Khushali': 'Khushal i', 'heitus': 'Leitus',
                  'cracksbecause': 'cracks because', 'Nautlius': 'Nautilus', 'trausted': 'trusted',
                  'Dardandus': 'Dardanus', 'Megatapirus': 'Mega tapirus', 'clusture': 'culture',
                  'vairamuthus': 'vairamuthu s', 'disclousre': 'disclosure',
                  'industrilaization': 'industrialization', 'musilms': 'muslims', 'Australia9': 'Australian',
                  'causinng': 'causing', 'ibdustries': 'industries', 'searious': 'serious',
                  'Coolmuster': 'Cool muster', 'sissyphus': 'sisyphus', ' justificatio ': 'justification',
                  'antihindus': 'anti hindus', 'Moduslink': 'Modus link', 'zymogenous': 'zymogen ous',
                  'prospeorus': 'prosperous', 'Retrocausality': 'Retro causality', 'FusionGPS': 'Fusion GPS',
                  'Mouseflow': 'Mouse flow', 'bootyplus': 'booty plus', 'Itylus': 'I tylus',
                  'Olnhausen': 'Olshausen', 'suspeect': 'suspect', 'entusiasta': 'enthusiast',
                  'fecetious': 'facetious', 'bussiest': 'fussiest', 'Draconius': 'Draconis',
                  'requsite': 'requisite', 'nauseatic': 'nausea tic', 'Brusssels': 'Brussels',
                  'repurcussion': 'repercussion', 'Jeisus': 'Jesus', 'philanderous': 'philander ous',
                  'muslisms': 'muslims', 'august2017': 'august 2017', 'calccalculus': 'calc calculus',
                  'unanonymously': 'un anonymously', 'Imaprtus': 'Impetus', 'carnivorus': 'carnivorous',
                  'Corypheus': 'Coryphees', 'austronauts': 'astronauts', 'neucleus': 'nucleus',
                  'housepoor': 'house poor', 'rescouses': 'responses', 'Tagushi': 'Tagus hi',
                  'hyperfocusing': 'hyper focusing', 'nutriteous': 'nutritious', 'chylus': 'chylous',
                  'preussure': 'pressure', 'outfocus': 'out focus', 'Hanfus': 'Hannus', 'Rustyrose': 'Rusty rose',
                  'vibhushant': 'vibhushan t', 'conciousnes': 'conciousness', 'Venus25': 'Venus',
                  'Sedataious': 'Seditious', 'promuslim': 'pro muslim', 'statusGuru': 'status Guru',
                  'yousician': 'musician', 'transgenus': 'trans genus', 'Pushbullet': 'Push bullet',
                  'jeesyllabus': 'jee syllabus', 'complusary': 'compulsory', 'Holocoust': 'Holocaust',
                  'careerplus': 'career plus', 'Lllustrate': 'Illustrate', 'Musino': 'Musion',
                  'Phinneus': 'Phineus', 'usedtoo': 'used too', 'JustBasic': 'Just Basic', 'webmusic': 'web music',
                  'TrustKit': 'Trust Kit', 'industrZgies': 'industries', 'rubustness': 'robustness',
                  'Missuses': 'Miss uses', 'Musturbation': 'Masturbation', 'bustees': 'bus tees',
                  'justyfy': 'justify', 'pegusus': 'pegasus', 'industrybuying': 'industry buying',
                  'advantegeous': 'advantageous', 'kotatsus': 'kotatsu s', 'justcreated': 'just created',
                  'simultameously': 'simultaneously', 'husoone': 'huso one', 'twiceusing': 'twice using',
                  'cetusplay': 'cetus play', 'sqamous': 'squamous', 'claustophobic': 'claustrophobic',
                  'Kaushika': 'Kaushik a', 'dioestrus': 'di oestrus', 'Degenerous': 'De generous',
                  'neculeus': 'nucleus', 'cutaneously': 'cu taneously', 'Alamotyrannus': 'Alamo tyrannus',
                  'Ivanious': 'Avanious', 'arceous': 'araceous', 'Flixbus': 'Flix bus', 'caausing': 'causing',
                  'publious': 'Publius', 'Juilus': 'Julius', 'Australianism': 'Australian ism',
                  'vetronus': 'verrons', 'nonspontaneous': 'non spontaneous', 'calcalus': 'calculus',
                  'commudus': 'Commodus', 'Rheusus': 'Rhesus', 'syallubus': 'syllabus', 'Yousician': 'Musician',
                  'qurush': 'qu rush', 'athiust': 'athirst', 'conclusionless': 'conclusion less',
                  'usertesting': 'user testing', 'redius': 'radius', 'Austrolia': 'Australia',
                  'sllaybus': 'syllabus', 'toponymous': 'top onymous', 'businiss': 'business',
                  'hyperthalamus': 'hyper thalamus', 'clause55': 'clause', 'cosicous': 'conscious',
                  'Sushena': 'Saphena', 'Luscinus': 'Luscious', 'Prussophile': 'Russophile', 'jeaslous': 'jealous',
                  'Austrelia': 'Australia', 'contiguious': 'contiguous',
                  'subconsciousnesses': 'sub consciousnesses', ' jusification ': 'justification',
                  'dilusion': 'delusion', 'anticoncussive': 'anti concussive', 'disngush': 'disgust',
                  'constiously': 'consciously', 'filabustering': 'filibustering', 'GAPbuster': 'GAP buster',
                  'insectivourous': 'insectivorous', 'glocuse': 'louse', 'Antritrust': 'Antitrust',
                  'thisAustralian': 'this Australian', 'FusionDrive': 'Fusion Drive', 'nuclus': 'nucleus',
                  'abussive': 'abusive', 'mustang1': 'mustangs', 'inradius': 'in radius', 'polonious': 'polonius',
                  'ofKulbhushan': 'of Kulbhushan', 'homosporous': 'homos porous', 'circumradius': 'circum radius',
                  'atlous': 'atrous', 'insustry': 'industry', 'campuswith': 'campus with', 'beacsuse': 'because',
                  'concuous': 'conscious', 'nonHindus': 'non Hindus', 'carnivourous': 'carnivorous',
                  'tradeplus': 'trade plus', 'Jeruselam': 'Jerusalem',
                  'musuclar': 'muscular', 'deangerous': 'dangerous', 'disscused': 'discussed',
                  'industdial': 'industrial', 'sallatious': 'fallacious', 'rohmbus': 'rhombus',
                  'golusu': 'gol usu', 'Minangkabaus': 'Minangkabau s', 'Mustansiriyah': 'Mustansiriya h',
                  'anomymously': 'anonymously', 'abonymously': 'anonymously', 'indrustry': 'industry',
                  'Musharrf': 'Musharraf', 'workouses': 'workhouses', 'sponataneously': 'spontaneously',
                  'anmuslim': 'an muslim', 'syallbus': 'syllabus', 'presumptuousnes': 'presumptuousness',
                  'Thaedus': 'Thaddus', 'industey': 'industry', 'hkust': 'hust', 'Kousseri': 'Kousser i',
                  'mousestats': 'mouses tats', 'russiagate': 'russia gate', 'simantaneously': 'simultaneously',
                  'Austertana': 'Auster tana', 'infussions': 'infusions', 'coclusion': 'conclusion',
                  'sustainabke': 'sustainable', 'tusami': 'tu sami', 'anonimously': 'anonymously',
                  'usebase': 'use base', 'balanoglossus': 'Balanoglossus', 'Unglaus': 'Ung laus',
                  'ignoramouses': 'ignoramuses', 'snuus': 'snugs', 'reusibility': 'reusability',
                  'Straussianism': 'Straussian ism', 'simoultaneously': 'simultaneously',
                  'realbonus': 'real bonus', 'nuchakus': 'nunchakus', 'annonimous': 'anonymous',
                  'Incestious': 'Incestuous', 'Manuscriptology': 'Manuscript ology', 'difusse': 'diffuse',
                  'Pliosaurus': 'Pliosaur us', 'cushelle': 'cush elle', 'Catallus': 'Catullus',
                  'MuscleBlaze': 'Muscle Blaze', 'confousing': 'confusing', 'enthusiasmless': 'enthusiasm less',
                  'Tetherusd': 'Tethered', 'Josephius': 'Josephus', 'jusrlt': 'just',
                  'simutaneusly': 'simultaneously', 'mountaneous': 'mountainous', 'Badonicus': 'Sardonicus',
                  'muccus': 'mucous', 'nicus': 'nidus', 'austinlizards': 'austin lizards',
                  'errounously': 'erroneously', 'Australua': 'Australia', 'sylaabus': 'syllabus',
                  'dusyant': 'distant', 'javadiscussion': 'java discussion', 'megabuses': 'mega buses',
                  'danergous': 'dangerous', 'contestious': 'contentious', 'exause': 'excuse',
                  'muscluar': 'muscular', 'avacous': 'vacuous', 'Ingenhousz': 'Ingenious',
                  'holocausting': 'holocaust ing', 'Pakustan': 'Pakistan', 'purusharthas': 'purushartha',
                  'bapus': 'bapu s', 'useul': 'useful', 'pretenious': 'pretentious', 'homogeneus': 'homogeneous',
                  'bhlushes': 'blushes', 'Saggittarius': 'Sagittarius', 'sportsusa': 'sports usa',
                  'kerataconus': 'keratoconus', 'infrctuous': 'infectuous', 'Anonoymous': 'Anonymous',
                  'triphosphorus': 'tri phosphorus', 'ridicjlously': 'ridiculously',
                  'worldbusiness': 'world business', 'hollcaust': 'holocaust', 'Dusra': 'Dura',
                  'meritious': 'meritorious', 'Sauskes': 'Causes', 'inudustry': 'industry',
                  'frustratd': 'frustrate', 'hypotenous': 'hypogenous', 'Dushasana': 'Dush asana',
                  'saadus': 'status', 'keratokonus': 'keratoconus', 'Jarrus': 'Harrus', 'neuseous': 'nauseous',
                  'simutanously': 'simultaneously', 'diphosphorus': 'di phosphorus', 'sulprus': 'surplus',
                  'Hasidus': 'Hasid us', 'suspenive': 'suspensive', 'illlustrator': 'illustrator',
                  'userflows': 'user flows', 'intrusivethoughts': 'intrusive thoughts', 'countinous': 'continuous',
                  'gpusli': 'gusli', 'Calculus1': 'Calculus', 'bushiri': 'Bushire',
                  'torvosaurus': 'Torosaurus', 'chestbusters': 'chest busters', 'Satannus': 'Sat annus',
                  'falaxious': 'fallacious', 'obnxious': 'obnoxious', 'tranfusions': 'transfusions',
                  'PlayMagnus': 'Play Magnus', 'Epicodus': 'Episodes', 'Hypercubus': 'Hypercubes',
                  'Musickers': 'Musick ers', 'programmebecause': 'programme because', 'indiginious': 'indigenous',
                  'housban': 'Housman', 'iusso': 'kusso', 'annilingus': 'anilingus', 'Nennus': 'Genius',
                  'pussboy': 'puss boy', 'Photoacoustics': 'Photo acoustics', 'Hindusthanis': 'Hindustanis',
                  'lndustrial': 'industrial', 'tyrannously': 'tyrannous', 'Susanoomon': 'Susanoo mon',
                  'colmbus': 'columbus', 'sussessful': 'successful', 'ousmania': 'ous mania',
                  'ilustrating': 'illustrating', 'famousbirthdays': 'famous birthdays',
                  'suspectance': 'suspect ance', 'extroneous': 'extraneous', 'teethbrush': 'teeth brush',
                  'abcmouse': 'abc mouse', 'degenerous': 'de generous', 'doesGauss': 'does Gauss',
                  'insipudus': 'insipidus', 'movielush': 'movie lush', 'Rustichello': 'Rustic hello',
                  'Firdausiya': 'Firdausi ya', 'checkusers': 'check users', 'householdware': 'household ware',
                  'prosporously': 'prosperously', 'SteLouse': 'Ste Louse', 'obfuscaton': 'obfuscation',
                  'amorphus': 'amorph us', 'trustworhy': 'trustworthy', 'celsious': 'cesious',
                  'dangorous': 'dangerous', 'anticancerous': 'anti cancerous', 'cousi ': 'cousin ',
                  'austroloid': 'australoid', 'fergussion': 'percussion', 'andKyokushin': 'and Kyokushin',
                  'cousan': 'cousin', 'Huskystar': 'Hu skystar', 'retrovisus': 'retrovirus', 'becausr': 'because',
                  'Jerusalsem': 'Jerusalem', 'motorious': 'notorious', 'industrilised': 'industrialised',
                  'powerballsusa': 'powerballs usa', 'monoceious': 'monoecious', 'batteriesplus': 'batteries plus',
                  'nonviscuous': 'nonviscous', 'industion': 'induction', 'bussinss': 'bussings',
                  'userbags': 'user bags', 'Jlius': 'Julius', 'thausand': 'thousand', 'plustwo': 'plus two',
                  'defpush': 'def push', 'subconcussive': 'sub concussive', 'muslium': 'muslim',
                  'industrilization': 'industrialization', 'Maurititus': 'Mauritius', 'uslme': 'some',
                  'Susgaon': 'Surgeon', 'Pantherous': 'Panther ous', 'antivirius': 'antivirus',
                  'Trustclix': 'Trust clix', 'silumtaneously': 'simultaneously', 'Icompus': 'Corpus',
                  'atonomous': 'autonomous', 'Reveuse': 'Reve use', 'legumnous': 'leguminous',
                  'syllaybus': 'syllabus', 'louspeaker': 'loudspeaker', 'susbtraction': 'substraction',
                  'virituous': 'virtuous', 'disastrius': 'disastrous', 'jerussalem': 'jerusalem',
                  'Industrailzed': 'Industrialized', 'recusion': 'recushion',
                  'simultenously': 'simultaneously',
                  'Pulphus': 'Pulpous', 'harbaceous': 'herbaceous', 'phlegmonous': 'phlegmon ous', 'use38': 'use',
                  'jusify': 'justify', 'instatanously': 'instantaneously', 'tetramerous': 'tetramer ous',
                  'usedvin': 'used vin', 'sagittarious': 'sagittarius', 'mausturbate': 'masturbate',
                  'subcautaneous': 'subcutaneous', 'dangergrous': 'dangerous', 'sylabbus': 'syllabus',
                  'hetorozygous': 'heterozygous', 'Ignasius': 'Ignacius', 'businessbor': 'business bor',
                  'Bhushi': 'Thushi', 'Moussolini': 'Mussolini', 'usucaption': 'usu caption',
                  'Customzation': 'Customization', 'cretinously': 'cretinous', 'genuiuses': 'geniuses',
                  'Moushmee': 'Mousmee', 'neigous': 'nervous',
                  'infrustructre': 'infrastructure', 'Ilusha': 'Ilesha', 'suconciously': 'unconciously',
                  'stusy': 'study', 'mustectomy': 'mastectomy', 'Farmhousebistro': 'Farmhouse bistro',
                  'instantanous': 'instantaneous', 'JustForex': 'Just Forex', 'Indusyry': 'Industry',
                  'mustabating': 'must abating', 'uninstrusive': 'unintrusive', 'customshoes': 'customs hoes',
                  'homageneous': 'homogeneous', 'Empericus': 'Imperious', 'demisexuality': 'demi sexuality',
                  'transexualism': 'transsexualism', 'sexualises': 'sexualise', 'demisexuals': 'demisexual',
                  'sexuly': 'sexily', 'Pornosexuality': 'Porno sexuality', 'sexond': 'second', 'sexxual': 'sexual',
                  'asexaul': 'asexual', 'sextactic': 'sex tactic', 'sexualityism': 'sexuality ism',
                  'monosexuality': 'mono sexuality', 'intwrsex': 'intersex', 'hypersexualize': 'hyper sexualize',
                  'homosexualtiy': 'homosexuality', 'examsexams': 'exams exams', 'sexmates': 'sex mates',
                  'sexyjobs': 'sexy jobs', 'sexitest': 'sexiest', 'fraysexual': 'fray sexual',
                  'sexsurrogates': 'sex surrogates', 'sexuallly': 'sexually', 'gamersexual': 'gamer sexual',
                  'greysexual': 'grey sexual', 'omnisexuality': 'omni sexuality', 'hetereosexual': 'heterosexual',
                  'productsexamples': 'products examples', 'sexgods': 'sex gods', 'semisexual': 'semi sexual',
                  'homosexulity': 'homosexuality', 'sexeverytime': 'sex everytime', 'neurosexist': 'neuro sexist',
                  'worldquant': 'world quant', 'Freshersworld': 'Freshers world', 'smartworld': 'sm artworld',
                  'Mistworlds': 'Mist worlds', 'boothworld': 'booth world', 'ecoworld': 'eco world',
                  'Ecoworld': 'Eco world', 'underworldly': 'under worldly', 'worldrank': 'world rank',
                  'Clearworld': 'Clear world', 'Boothworld': 'Booth world', 'Rimworld': 'Rim world',
                  'cryptoworld': 'crypto world', 'machineworld': 'machine world', 'worldwideley': 'worldwide ley',
                  'capuletwant': 'capulet want', 'Bhagwanti': 'Bhagwant i', 'Unwanted72': 'Unwanted 72',
                  'wantrank': 'want rank',
                  'willhappen': 'will happen', 'thateasily': 'that easily',
                  'Whatevidence': 'What evidence', 'metaphosphates': 'meta phosphates',
                  'exilarchate': 'exilarch ate', 'aulphate': 'sulphate', 'Whateducation': 'What education',
                  'persulphates': 'per sulphates', 'disulphate': 'di sulphate', 'picosulphate': 'pico sulphate',
                  'tetraosulphate': 'tetrao sulphate', 'prechinese': 'pre chinese',
                  'Hellochinese': 'Hello chinese', 'muchdeveloped': 'much developed', 'stomuch': 'stomach',
                  'Whatmakes': 'What makes', 'Lensmaker': 'Lens maker', 'eyemake': 'eye make',
                  'Techmakers': 'Tech makers', 'cakemaker': 'cake maker', 'makeup411': 'makeup 411',
                  'objectmake': 'object make', 'crazymaker': 'crazy maker', 'techmakers': 'tech makers',
                  'makedonian': 'macedonian', 'makeschool': 'make school', 'anxietymake': 'anxiety make',
                  'makeshifter': 'make shifter', 'countryball': 'country ball', 'Whichcountry': 'Which country',
                  'countryHow': 'country How', 'Zenfone': 'Zen fone', 'Electroneum': 'Electro neum',
                  'electroneum': 'electro neum', 'Demonetisation': 'demonetization', 'zenfone': 'zen fone',
                  'ZenFone': 'Zen Fone', 'onecoin': 'one coin', 'demonetizing': 'demonetized',
                  'iphone7': 'iPhone', 'iPhone6': 'iPhone', 'microneedling': 'micro needling', 'iphone6': 'iPhone',
                  'Monegasques': 'Monegasque s', 'demonetised': 'demonetized',
                  'EveryoneDiesTM': 'EveryoneDies TM', 'teststerone': 'testosterone', 'DoneDone': 'Done Done',
                  'papermoney': 'paper money', 'Sasabone': 'Sasa bone', 'Blackphone': 'Black phone',
                  'Bonechiller': 'Bone chiller', 'Moneyfront': 'Money front', 'workdone': 'work done',
                  'iphoneX': 'iPhone', 'roxycodone': 'r oxycodone',
                  'moneycard': 'money card', 'Fantocone': 'Fantocine', 'eletronegativity': 'electronegativity',
                  'mellophones': 'mellophone s', 'isotones': 'iso tones', 'donesnt': 'doesnt',
                  'thereanyone': 'there anyone', 'electronegativty': 'electronegativity',
                  'commissiioned': 'commissioned', 'earvphone': 'earphone', 'condtioners': 'conditioners',
                  'demonetistaion': 'demonetization', 'ballonets': 'ballo nets', 'DoneClaim': 'Done Claim',
                  'alimoney': 'alimony', 'iodopovidone': 'iodo povidone', 'bonesetters': 'bone setters',
                  'componendo': 'compon endo', 'probationees': 'probationers', 'one300': 'one 300',
                  'nonelectrolyte': 'non electrolyte', 'ozonedepletion': 'ozone depletion',
                  'Stonehart': 'Stone hart', 'Vodafone2': 'Vodafones', 'chaparone': 'chaperone',
                  'Noonein': 'Noo nein', 'Frosione': 'Erosion', 'IPhone7': 'Iphone', 'pentanone': 'penta none',
                  'poneglyphs': 'pone glyphs', 'cyclohexenone': 'cyclohexanone', 'marlstone': 'marls tone',
                  'androneda': 'andromeda', 'iphone8': 'iPhone', 'acidtone': 'acid tone',
                  'noneconomically': 'non economically', 'Honeyfund': 'Honey fund', 'germanophone': 'Germanophobe',
                  'Democratizationed': 'Democratization ed', 'haoneymoon': 'honeymoon', 'iPhone7': 'iPhone 7',
                  'someonewith': 'some onewith', 'Hexanone': 'Hexa none', 'bonespur': 'bones pur',
                  'sisterzoned': 'sister zoned', 'HasAnyone': 'Has Anyone',
                  'stonepelters': 'stone pelters', 'Chronexia': 'Chronaxia', 'brotherzone': 'brother zone',
                  'brotherzoned': 'brother zoned', 'fonecare': 'f onecare', 'nonexsistence': 'nonexistence',
                  'conents': 'contents', 'phonecases': 'phone cases', 'Commissionerates': 'Commissioner ates',
                  'activemoney': 'active money', 'dingtone': 'ding tone', 'wheatestone': 'wheatstone',
                  'chiropractorone': 'chiropractor one', 'heeadphones': 'headphones', 'Maimonedes': 'Maimonides',
                  'onepiecedeals': 'onepiece deals', 'oneblade': 'one blade', 'venetioned': 'Venetianed',
                  'sunnyleone': 'sunny leone', 'prendisone': 'prednisone', 'Anglosaxophone': 'Anglo saxophone',
                  'Blackphones': 'Black phones', 'jionee': 'jinnee', 'chromonema': 'chromo nema',
                  'iodoketones': 'iodo ketones', 'demonetizations': 'demonetization', 'aomeone': 'someone',
                  'trillonere': 'trillones', 'abandonee': 'abandon',
                  'MasterColonel': 'Master Colonel', 'fronend': 'friend', 'Wildstone': 'Wilds tone',
                  'patitioned': 'petitioned', 'lonewolfs': 'lone wolfs', 'Spectrastone': 'Spectra stone',
                  'dishonerable': 'dishonorable', 'poisiones': 'poisons',
                  'condioner': 'conditioner', 'unpermissioned': 'unper missioned', 'friedzone': 'fried zone',
                  'umumoney': 'umu money', 'anyonestudied': 'anyone studied', 'dictioneries': 'dictionaries',
                  'nosebone': 'nose bone', 'ofVodafone': 'of Vodafone',
                  'Yumstone': 'Yum stone', 'oxandrolonesteroid': 'oxandrolone steroid',
                  'Mifeprostone': 'Mifepristone', 'pheramones': 'pheromones',
                  'sinophone': 'Sinophobe', 'peloponesian': 'peloponnesian', 'michrophone': 'microphone',
                  'commissionets': 'commissioners', 'methedone': 'methadone', 'cobditioners': 'conditioners',
                  'urotone': 'protone', 'smarthpone': 'smartphone', 'conecTU': 'connect you', 'beloney': 'boloney',
                  'comfortzone': 'comfort zone', 'testostersone': 'testosterone', 'camponente': 'component',
                  'Idonesia': 'Indonesia', 'dolostones': 'dolostone', 'psiphone': 'psi phone',
                  'ceftriazone': 'ceftriaxone', 'feelonely': 'feel onely', 'monetation': 'moderation',
                  'activationenergy': 'activation energy', 'moneydriven': 'money driven',
                  'staionery': 'stationery', 'zoneflex': 'zone flex', 'moneycash': 'money cash',
                  'conectiin': 'connection', 'Wannaone': 'Wanna one',
                  'Pictones': 'Pict ones', 'demonentization': 'demonetization',
                  'phenonenon': 'phenomenon', 'evenafter': 'even after', 'Sevenfriday': 'Seven friday',
                  'Devendale': 'Evendale', 'theeventchronicle': 'the event chronicle',
                  'seventysomething': 'seventy something', 'sevenpointed': 'seven pointed',
                  'richfeel': 'rich feel', 'overfeel': 'over feel', 'feelingstupid': 'feeling stupid',
                  'Photofeeler': 'Photo feeler', 'feelomgs': 'feelings', 'feelinfs': 'feelings',
                  'PlayerUnknown': 'Player Unknown', 'Playerunknown': 'Player unknown', 'knowlefge': 'knowledge',
                  'knowledgd': 'knowledge', 'knowledeg': 'knowledge', 'knowble': 'Knowle', 'Howknow': 'Howk now',
                  'knowledgeWoods': 'knowledge Woods', 'knownprogramming': 'known programming',
                  'selfknowledge': 'self knowledge', 'knowldage': 'knowledge', 'knowyouve': 'know youve',
                  'aknowlege': 'knowledge', 'Audetteknown': 'Audette known', 'knowlegdeable': 'knowledgeable',
                  'trueoutside': 'true outside', 'saynthesize': 'synthesize', 'EssayTyper': 'Essay Typer',
                  'meesaya': 'mee saya', 'Rasayanam': 'Rasayan am', 'fanessay': 'fan essay', 'momsays': 'moms ays',
                  'sayying': 'saying', 'saydaw': 'say daw', 'Fanessay': 'Fan essay', 'theyreally': 'they really',
                  'gayifying': 'gayed up with homosexual love', 'gayke': 'gay Online retailers',
                  'Lingayatism': 'Lingayat',
                  'macapugay': 'Macaulay', 'jewsplain': 'jews plain',
                  'banggood': 'bang good', 'goodfriends': 'good friends',
                  'goodfirms': 'good firms', 'Banggood': 'Bang good', 'dogooder': 'do gooder',
                  'stillshots': 'stills hots', 'stillsuits': 'still suits', 'panromantic': 'pan romantic',
                  'paracommando': 'para commando', 'romantize': 'romanize', 'manupulative': 'manipulative',
                  'manjha': 'mania', 'mankrit': 'mank rit',
                  'heteroromantic': 'hetero romantic', 'pulmanery': 'pulmonary', 'manpads': 'man pads',
                  'supermaneuverable': 'super maneuverable', 'mandatkry': 'mandatory', 'armanents': 'armaments',
                  'manipative': 'mancipative', 'himanity': 'humanity', 'maneuever': 'maneuver',
                  'Kumarmangalam': 'Kumar mangalam', 'Brahmanwadi': 'Brahman wadi',
                  'exserviceman': 'ex serviceman',
                  'managewp': 'managed', 'manies': 'many', 'recordermans': 'recorder mans',
                  'Feymann': 'Heymann', 'salemmango': 'salem mango', 'manufraturing': 'manufacturing',
                  'sreeman': 'freeman', 'tamanaa': 'Tamanac', 'chlamydomanas': 'chlamydomonas',
                  'comandant': 'commandant', 'huemanity': 'humanity', 'manaagerial': 'managerial',
                  'lithromantics': 'lith romantics',
                  'geramans': 'germans', 'Nagamandala': 'Naga mandala', 'humanitariarism': 'humanitarianism',
                  'wattman': 'watt man', 'salesmanago': 'salesman ago', 'Washwoman': 'Wash woman',
                  'rammandir': 'ram mandir', 'nomanclature': 'nomenclature', 'Haufman': 'Kaufman',
                  'prefomance': 'performance', 'ramanunjan': 'Ramanujan', 'Freemansonry': 'Freemasonry',
                  'supermaneuverability': 'super maneuverability', 'manstruate': 'menstruate',
                  'Tarumanagara': 'Taruma nagara', 'RomanceTale': 'Romance Tale', 'heteromantic': 'hete romantic',
                  'terimanals': 'terminals', 'womansplaining': 'wo mansplaining',
                  'performancelearning': 'performance learning', 'sociomantic': 'sciomantic',
                  'batmanvoice': 'batman voice', 'PerformanceTesting': 'Performance Testing',
                  'manorialism': 'manorial ism', 'newscommando': 'news commando',
                  'Entwicklungsroman': 'Entwicklungs roman',
                  'Kunstlerroman': 'Kunstler roman', 'bodhidharman': 'Bodhidharma', 'Howmaney': 'How maney',
                  'manufucturing': 'manufacturing', 'remmaning': 'remaining', 'rangeman': 'range man',
                  'mythomaniac': 'mythomania', 'katgmandu': 'katmandu',
                  'Superowoman': 'Superwoman', 'Rahmanland': 'Rahman land', 'Dormmanu': 'Dormant',
                  'Geftman': 'Gentman', 'manufacturig': 'manufacturing', 'bramanistic': 'Brahmanistic',
                  'padmanabhanagar': 'padmanabhan agar', 'homoromantic': 'homo romantic', 'femanists': 'feminists',
                  'demihuman': 'demi human', 'manrega': 'Manresa', 'Pasmanda': 'Pas manda',
                  'manufacctured': 'manufactured', 'remaninder': 'remainder', 'Marimanga': 'Mari manga',
                  'Sloatman': 'Sloat man', 'manlet': 'man let', 'perfoemance': 'performance',
                  'mangolian': 'mongolian', 'mangekyu': 'mange kyu', 'mansatory': 'mandatory',
                  'managemebt': 'management', 'manufctures': 'manufactures', 'Bramanical': 'Brahmanical',
                  'manaufacturing': 'manufacturing', 'Lakhsman': 'Lakhs man', 'Sarumans': 'Sarum ans',
                  'mangalasutra': 'mangalsutra', 'Germanised': 'German ised',
                  'managersworking': 'managers working', 'cammando': 'commando', 'mandrillaris': 'mandrill aris',
                  'Emmanvel': 'Emmarvel', 'manupalation': 'manipulation', 'welcomeromanian': 'welcome romanian',
                  'humanfemale': 'human female', 'mankirt': 'mankind', 'Haffmann': 'Hoffmann',
                  'Panromantic': 'Pan romantic', 'demantion': 'detention', 'Suparwoman': 'Superwoman',
                  'parasuramans': 'parasuram ans', 'sulmann': 'Suilmann', 'Shubman': 'Subman',
                  'manspread': 'man spread', 'mandingan': 'Mandingan', 'mandalikalu': 'mandalika lu',
                  'manufraturer': 'manufacturer', 'Wedgieman': 'Wedgie man', 'manwues': 'manages',
                  'humanzees': 'human zees', 'Steymann': 'Stedmann', 'Jobberman': 'Jobber man',
                  'maniquins': 'mani quins', 'biromantical': 'bi romantical', 'Rovman': 'Roman',
                  'pyromantic': 'pyro mantic', 'Tastaman': 'Rastaman', 'Spoolman': 'Spool man',
                  'Subramaniyan': 'Subramani yan', 'abhimana': 'abhiman a', 'manholding': 'man holding',
                  'seviceman': 'serviceman', 'womansplained': 'womans plained', 'manniya': 'mania',
                  'Bhraman': 'Braman', 'Laakman': 'Layman', 'mansturbate': 'masturbate',
                  'Sulamaniya': 'Sulamani ya', 'demanters': 'decanters', 'postmanare': 'postman are',
                  'mannualy': 'annual', 'rstman': 'Rotman', 'permanentjobs': 'permanent jobs',
                  'Allmang': 'All mang', 'TradeCommander': 'Trade Commander', 'BasedStickman': 'Based Stickman',
                  'Deshabhimani': 'Desha bhimani', 'manslamming': 'mans lamming', 'Brahmanwad': 'Brahman wad',
                  'fundemantally': 'fundamentally', 'supplemantary': 'supplementary', 'egomanias': 'ego manias',
                  'manvantar': 'Manvantara', 'spymania': 'spy mania', 'mangonada': 'mango nada',
                  'manthras': 'mantras', 'Humanpark': 'Human park', 'manhuas': 'mahuas',
                  'manterrupting': 'interrupting', 'dermatillomaniac': 'dermatillomania',
                  'performancies': 'performances', 'manipulant': 'manipulate',
                  'painterman': 'painter man', 'mangalik': 'manglik',
                  'Neurosemantics': 'Neuro semantics', 'discrimantion': 'discrimination',
                  'Womansplaining': 'feminist', 'mongodump': 'mongo dump', 'roadgods': 'road gods',
                  'Oligodendraglioma': 'Oligodendroglioma', 'unrightly': 'un rightly', 'Janewright': 'Jane wright',
                  ' righten ': ' tighten ', 'brightiest': 'brightest',
                  'frighter': 'fighter', 'righteouness': 'righteousness', 'triangleright': 'triangle right',
                  'Brightspace': 'Brights pace', 'techinacal': 'technical', 'chinawares': 'china wares',
                  'Vancouever': 'Vancouver', 'cheverlet': 'cheveret', 'deverstion': 'diversion',
                  'everbodys': 'everybody', 'Dramafever': 'Drama fever', 'reverificaton': 'reverification',
                  'canterlever': 'canter lever', 'keywordseverywhere': 'keywords everywhere',
                  'neverunlearned': 'never unlearned', 'everyfirst': 'every first',
                  'neverhteless': 'nevertheless', 'clevercoyote': 'clever coyote', 'irrevershible': 'irreversible',
                  'achievership': 'achievers hip', 'easedeverything': 'eased everything', 'youbever': 'you bever',
                  'everperson': 'ever person', 'everydsy': 'everyday', 'whemever': 'whenever',
                  'everyonr': 'everyone', 'severiity': 'severity', 'narracist': 'nar racist',
                  'racistly': 'racist', 'takesuch': 'take such', 'mystakenly': 'mistakenly',
                  'shouldntake': 'shouldnt take', 'Kalitake': 'Kali take', 'msitake': 'mistake',
                  'straitstimes': 'straits times', 'timefram': 'timeframe', 'watchtime': 'watch time',
                  'timetraveling': 'timet raveling', 'peactime': 'peacetime', 'timetabe': 'timetable',
                  'cooktime': 'cook time', 'blocktime': 'block time', 'timesjobs': 'times jobs',
                  'timesence': 'times ence', 'Touchtime': 'Touch time', 'timeloop': 'time loop',
                  'subcentimeter': 'sub centimeter', 'timejobs': 'time jobs', 'Guardtime': 'Guard time',
                  'realtimepolitics': 'realtime politics', 'loadingtimes': 'loading times',
                  'timesnow': '24-hour English news channel in India', 'timesspark': 'times spark',
                  'timetravelling': 'timet ravelling',
                  'antimeter': 'anti meter', 'timewaste': 'time waste', 'cryptochristians': 'crypto christians',
                  'Whatcould': 'What could', 'becomesdouble': 'becomes double', 'deathbecomes': 'death becomes',
                  'youbecome': 'you become', 'greenseer': 'people who possess the magical ability',
                  'rseearch': 'research', 'homeseek': 'home seek',
                  'Greenseer': 'people who possess the magical ability', 'starseeders': 'star seeders',
                  'seekingmillionaire': 'seeking millionaire', 'see\u202c': 'see',
                  'seeies': 'series', 'CodeAgon': 'Code Agon',
                  'royago': 'royal', 'Dragonkeeper': 'Dragon keeper', 'mcgreggor': 'McGregor',
                  'catrgory': 'category', 'Dragonknight': 'Dragon knight', 'Antergos': 'Anteros',
                  'togofogo': 'togo fogo', 'mongorestore': 'mongo restore', 'gorgops': 'gorgons',
                  'withgoogle': 'with google', 'goundar': 'Gondar', 'algorthmic': 'algorithmic',
                  'goatnuts': 'goat nuts', 'vitilgo': 'vitiligo', 'polygony': 'poly gony',
                  'digonals': 'diagonals', 'Luxemgourg': 'Luxembourg', 'UCSanDiego': 'UC SanDiego',
                  'Ringostat': 'Ringo stat', 'takingoff': 'taking off', 'MongoImport': 'Mongo Import',
                  'alggorithms': 'algorithms', 'dragonknight': 'dragon knight', 'negotiatior': 'negotiation',
                  'gomovies': 'go movies', 'Withgott': 'Without',
                  'categoried': 'categories', 'Stocklogos': 'Stock logos', 'Pedogogical': 'Pedological',
                  'Wedugo': 'Wedge', 'golddig': 'gold dig', 'goldengroup': 'golden group',
                  'merrigo': 'merligo', 'googlemapsAPI': 'googlemaps API', 'goldmedal': 'gold medal',
                  'golemized': 'polemized', 'Caligornia': 'California', 'unergonomic': 'un ergonomic',
                  'fAegon': 'wagon', 'vertigos': 'vertigo s', 'trigonomatry': 'trigonometry',
                  'hypogonadic': 'hypogonadia', 'Mogolia': 'Mongolia', 'governmaent': 'government',
                  'ergotherapy': 'ergo therapy', 'Bogosort': 'Bogo sort', 'goalwise': 'goal wise',
                  'alogorithms': 'algorithms', 'MercadoPago': 'Mercado Pago', 'rivigo': 'rivi go',
                  'govshutdown': 'gov shutdown', 'gorlfriend': 'girlfriend',
                  'stategovt': 'state govt', 'Chickengonia': 'Chicken gonia', 'Yegorovich': 'Yegorov ich',
                  'regognitions': 'recognitions', 'gorichen': 'Gori Chen Mountain',
                  'goegraphies': 'geographies', 'gothras': 'goth ras', 'belagola': 'bela gola',
                  'snapragon': 'snapdragon', 'oogonial': 'oogonia l', 'Amigofoods': 'Amigo foods',
                  'Sigorn': 'son of Styr', 'algorithimic': 'algorithmic',
                  'innermongolians': 'inner mongolians', 'ArangoDB': 'Arango DB', 'zigolo': 'gigolo',
                  'regognized': 'recognized', 'Moongot': 'Moong ot', 'goldquest': 'gold quest',
                  'catagorey': 'category', 'got7': 'got', 'jetbingo': 'jet bingo', 'Dragonchain': 'Dragon chain',
                  'catwgorized': 'categorized', 'gogoro': 'gogo ro', 'Tobagoans': 'Tobago ans',
                  'digonal': 'di gonal', 'algoritmic': 'algorismic', 'dragonflag': 'dragon flag',
                  'Indigoflight': 'Indigo flight',
                  'governening': 'governing', 'ergosphere': 'ergo sphere',
                  'pingo5': 'pingo', 'Montogo': 'montego', 'Rivigo': 'technology-enabled logistics company',
                  'Jigolo': 'Gigolo', 'phythagoras': 'pythagoras', 'Mangolian': 'Mongolian',
                  'forgottenfaster': 'forgotten faster', 'stargold': 'a Hindi movie channel',
                  'googolplexain': 'googolplexian', 'corpgov': 'corp gov',
                  'govtribe': 'provides real-time federal contracting market intel',
                  'dragonglass': 'dragon glass', 'gorakpur': 'Gorakhpur', 'MangoPay': 'Mango Pay',
                  'chigoe': 'sub-tropical climates', 'BingoBox': 'an investment company', '走go': 'go',
                  'followingorder': 'following order', 'pangolinminer': 'pangolin miner',
                  'negosiation': 'negotiation', 'lexigographers': 'lexicographers', 'algorithom': 'algorithm',
                  'unforgottable': 'unforgettable', 'wellsfargoemail': 'wellsfargo email',
                  'daigonal': 'diagonal', 'Pangoro': 'cantankerous Pokemon', 'negotiotions': 'negotiations',
                  'Swissgolden': 'Swiss golden', 'google4': 'google', 'Agoraki': 'Ago raki',
                  'Garthago': 'Carthago', 'Stegosauri': 'stegosaurus', 'ergophobia': 'ergo phobia',
                  'bigolive': 'big olive', 'bittergoat': 'bitter goat', 'naggots': 'faggots',
                  'googology': 'online encyclopedia', 'algortihms': 'algorithms', 'bengolis': 'Bengalis',
                  'fingols': 'Finnish people are supposedly descended from Mongols',
                  'savethechildren': 'save thechildren',
                  'stopings': 'stoping', 'stopsits': 'stop sits', 'stopsigns': 'stop signs',
                  'Galastop': 'Galas top', 'pokestops': 'pokes tops', 'forcestop': 'forces top',
                  'Hopstop': 'Hops top', 'stoppingexercises': 'stopping exercises', 'coinstop': 'coins top',
                  'stoppef': 'stopped', 'workaway': 'work away', 'snazzyway': 'snazzy way',
                  'Rewardingways': 'Rewarding ways', 'cloudways': 'cloud ways', 'Cloudways': 'Cloud ways',
                  'Brainsway': 'Brains way', 'nesraway': 'nearaway',
                  'AlwaysHired': 'Always Hired', 'expessway': 'expressway', 'Syncway': 'Sync way',
                  'LeewayHertz': 'Blockchain Company', 'towayrds': 'towards', 'swayable': 'sway able',
                  'Telloway': 'Tello way', 'palsmodium': 'plasmodium', 'Gobackmodi': 'Goback modi',
                  'comodies': 'corodies', 'islamphobic': 'islam phobic', 'islamphobia': 'islam phobia',
                  'citiesbetter': 'cities better', 'betterv3': 'better', 'betterDtu': 'better Dtu',
                  'Babadook': 'a horror drama film', 'Ahemadabad': 'Ahmadabad', 'faidabad': 'Faizabad',
                  'Amedabad': 'Ahmedabad', 'kabadii': 'kabaddi', 'badmothing': 'badmouthing',
                  'badminaton': 'badminton', 'badtameezdil': 'badtameez dil', 'badeffects': 'bad effects',
                  '∠bad': 'bad', 'ahemadabad': 'Ahmadabad', 'embaded': 'embased', 'Isdhanbad': 'Is dhanbad',
                  'badgermoles': 'enormous, blind mammal', 'allhabad': 'Allahabad', 'ghazibad': 'ghazi bad',
                  'htderabad': 'Hyderabad', 'Auragabad': 'Aurangabad', 'ahmedbad': 'Ahmedabad',
                  'ahmdabad': 'Ahmadabad', 'alahabad': 'Allahabad',
                  'Hydeabad': 'Hyderabad', 'Gyroglove': 'wearable technology', 'foodlovee': 'food lovee',
                  'slovenised': 'slovenia', 'handgloves': 'hand gloves', 'lovestep': 'love step',
                  'lovejihad': 'love jihad', 'RolloverBox': 'Rollover Box', 'stupidedt': 'stupidest',
                  'toostupid': 'too stupid',
                  'pakistanisbeautiful': 'pakistanis beautiful', 'ispakistan': 'is pakistan',
                  'inpersonations': 'impersonations', 'medicalperson': 'medical person',
                  'interpersonation': 'inter personation', 'workperson': 'work person',
                  'personlich': 'person lich', 'persoenlich': 'person lich',
                  'middleperson': 'middle person', 'personslized': 'personalized',
                  'personifaction': 'personification', 'welcomemarriage': 'welcome marriage',
                  'come2': 'come to', 'upcomedians': 'up comedians', 'overvcome': 'overcome',
                  'talecome': 'tale come', 'cometitive': 'competitive', 'arencome': 'aren come',
                  'achecomes': 'ache comes', '」come': 'come',
                  'comepleted': 'completed', 'overcomeanxieties': 'overcome anxieties',
                  'demigirl': 'demi girl', 'gridgirl': 'female models of the race', 'halfgirlfriend': 'half girlfriend',
                  'girlriend': 'girlfriend', 'fitgirl': 'fit girl', 'girlfrnd': 'girlfriend', 'awrong': 'aw rong',
                  'northcap': 'north cap', 'productionsupport': 'production support',
                  'Designbold': 'Online Photo Editor Design Studio',
                  'skyhold': 'sky hold', 'shuoldnt': 'shouldnt', 'anarold': 'Android', 'yaerold': 'year old',
                  'soldiders': 'soldiers', 'indrold': 'Android', 'blindfoldedly': 'blindfolded',
                  'overcold': 'over cold', 'Goldmont': 'microarchitecture in Intel', 'boldspot': 'bolds pot',
                  'Rankholders': 'Rank holders', 'cooldrink': 'cool drink', 'beltholders': 'belt holders',
                  'GoldenDict': 'open-source dictionary program', 'softskill': 'softs kill',
                  'Cooldige': 'the 30th president of the United States',
                  'newkiller': 'new killer', 'skillselect': 'skills elect', 'nonskilled': 'non skilled',
                  'killyou': 'kill you', 'Skillport': 'Army e-Learning Program', 'unkilled': 'un killed',
                  'killikng': 'killing', 'killograms': 'kilograms',
                  'Worldkillers': 'World killers', 'reskilled': 'skilled',
                  'killedshivaji': 'killed shivaji', 'honorkillings': 'honor killings',
                  'skillclasses': 'skill classes', 'microskills': 'micros kills',
                  'Skillselect': 'Skills elect', 'ratkill': 'rat kill',
                  'pleasegive': 'please give', 'flashgive': 'flash give',
                  'southerntelescope': 'southern telescope', 'westsouth': 'west south',
                  'southAfricans': 'south Africans', 'Joboutlooks': 'Job outlooks', 'joboutlook': 'job outlook',
                  'Outlook365': 'Outlook 365', 'Neulife': 'Neu life', 'qualifeid': 'qualified',
                  'nullifed': 'nullified', 'lifeaffect': 'life affect', 'lifestly': 'lifestyle',
                  'aristocracylifestyle': 'aristocracy lifestyle', 'antilife': 'anti life',
                  'afterafterlife': 'after afterlife', 'lifestylye': 'lifestyle', 'prelife': 'pre life',
                  'lifeute': 'life ute', 'liferature': 'literature',
                  'securedlife': 'secured life', 'doublelife': 'double life', 'antireligion': 'anti religion',
                  'coreligionist': 'co religionist', 'petrostates': 'petro states', 'otherstates': 'others tates',
                  'spacewithout': 'space without', 'withoutyou': 'without you',
                  'withoutregistered': 'without registered', 'weightwithout': 'weight without',
                  'withoutcheck': 'without check', 'milkwithout': 'milk without',
                  'Highschoold': 'High school', 'memoney': 'money', 'moneyof': 'mony of', 'Oneplus': 'OnePlus',
                  'OnePlus': 'Chinese smartphone manufacturer', 'Beerus': 'the God of Destruction',
                  'takeoverr': 'takeover', 'demonetizedd': 'demonetized', 'polyhouse': 'Polytunnel',
                  'Elitmus': 'eLitmus', 'eLitmus': 'Indian company that helps companies in hiring employees',
                  'becone': 'become', 'nestaway': 'nest away', 'takeoverrs': 'takeovers', 'Istop': 'I stop',
                  'Austira': 'Australia', 'germeny': 'Germany', 'mansoon': 'man soon',
                  'worldmax': 'wholesaler of drum parts',
                  'ammusement': 'amusement', 'manyare': 'many are', 'supplymentary': 'supply mentary',
                  'timesup': 'times up', 'homologus': 'homologous', 'uimovement': 'ui movement', 'spause': 'spouse',
                  'aesexual': 'asexual', 'Iovercome': 'I overcome', 'developmeny': 'development',
                  'hindusm': 'hinduism', 'sexpat': 'sex tourism', 'sunstop': 'sun stop', 'polyhouses': 'Polytunnel',
                  'usefl': 'useful', 'Fundamantal': 'fundamental', 'environmentai': 'environmental',
                  'Redmi': 'Xiaomi Mobile', 'Loy Machedo': ' Motivational Speaker ', 'unacademy': 'Unacademy',
                  'Boruto': 'Naruto Next Generations', 'Upwork': 'Up work',
                  'Unacademy': 'educational technology company',
                  'HackerRank': 'Hacker Rank', 'upwork': 'up work', 'Chromecast': 'Chrome cast',
                  'microservices': 'micro services', 'Undertale': 'video game', 'undergraduation': 'under graduation',
                  'chapterwise': 'chapter wise', 'twinflame': 'twin flame', 'Hotstar': 'Hot star',
                  'blockchains': 'blockchain',
                  'darkweb': 'dark web', 'Microservices': 'Micro services', 'Nearbuy': 'Nearby',
                  ' Padmaavat ': ' Padmavati ', ' padmavat ': ' Padmavati ', ' Padmaavati ': ' Padmavati ',
                  ' Padmavat ': ' Padmavati ', ' internshala ': ' internship and online training platform in India ',
                  'dream11': ' fantasy sports platform in India ', 'conciousnesss': 'consciousnesses',
                  'Dream11': ' fantasy sports platform in India ', 'cointry': 'country', ' coinvest ': ' invest ',
                  '23 andme': 'privately held personal genomics and biotechnology company in California',
                  'Trumpism': 'philosophy and politics espoused by Donald Trump',
                  'Trumpian': 'viewpoints of President Donald Trump', 'Trumpists': 'admirer of Donald Trump',
                  'coincidents': 'coincidence', 'coinsized': 'coin sized', 'coincedences': 'coincidences',
                  'cointries': 'countries', 'coinsidered': 'considered', 'coinfirm': 'confirm',
                  'humilates':'humiliates', 'vicevice':'vice vice', 'politicak':'political', 'Sumaterans':'Sumatrans',
                  'Kamikazis':'Kamikazes', 'unmoraled':'unmoral', 'eduacated':'educated', 'moraled':'morale',
                  'Amharc':'Amarc', 'where Burkhas':'wear Burqas', 'Baloochistan':'Balochistan', 'durgahs':'durgans',
                  'illigitmate':'illegitimate', 'hillum':'helium','treatens':'threatens','mutiliating':'mutilating',
                  'speakingly':'speaking', 'pretex':'pretext', 'menstruateion':'menstruation', 
                  'genocidizing':'genociding', 'maratis':'Maratism','Parkistinian':'Pakistani', 'SPEICIAL':'SPECIAL',
                  'REFERNECE':'REFERENCE', 'provocates':'provokes', 'FAMINAZIS':'FEMINAZIS', 'repugicans':'republicans',
                  'tonogenesis':'tone', 'winor':'win', 'redicules':'ridiculous', 'Beluchistan':'Balochistan', 
                  'volime':'volume', 'namaj':'namaz', 'CONgressi':'Congress', 'Ashifa':'Asifa', 'queffing':'queefing',
                  'montheistic':'nontheistic', 'Rajsthan':'Rajasthan', 'Rajsthanis':'Rajasthanis', 'specrum':'spectrum',
                  'brophytes':'bryophytes', 'adhaar':'Adhara', 'slogun':'slogan', 'harassd':'harassed',
                  'transness':'trans gender', 'Insdians':'Indians', 'Trampaphobia':'Trump aphobia', 'attrected':'attracted',
                  'Yahtzees':'Yahtzee', 'thiests':'atheists', 'thrir':'their', 'extraterestrial':'extraterrestrial',
                  'silghtest':'slightest', 'primarty':'primary','brlieve':'believe', 'fondels':'fondles',
                  'loundly':'loudly', 'bootythongs':'booty thongs', 'understamding':'understanding', 'degenarate':'degenerate',
                  'narsistic':'narcistic', 'innerskin':'inner skin','spectulated':'speculated', 'hippocratical':'Hippocratical',
                  'itstead':'instead', 'parralels':'parallels', 'sloppers':'slippers'
                  }

def clean_bad_case_words(text):
    for bad_word in bad_case_words:
        if bad_word in text:
            text = text.replace(bad_word, bad_case_words[bad_word])
    return text

mis_connect_list = ['(W|w)hat', '(W|w)hy', '(H|h)ow', '(W|w)hich', '(W|w)here', '(W|w)ill']
mis_connect_re = re.compile('(%s)' % '|'.join(mis_connect_list))

mis_spell_mapping = {'whattsup': 'WhatsApp', 'whatasapp':'WhatsApp', 'whatsupp':'WhatsApp', 
                      'whatcus':'what cause', 'arewhatsapp': 'are WhatsApp', 'Hwhat':'what',
                      'Whwhat': 'What', 'whatshapp':'WhatsApp', 'howhat':'how that',
                      # why
                      'Whybis':'Why is', 'laowhy86':'Foreigners who do not respect China',
                      'Whyco-education':'Why co-education',
                      # How
                      "Howddo":"How do", 'Howeber':'However', 'Showh':'Show',
                      "Willowmagic":'Willow magic', 'WillsEye':'Will Eye', 'Williby':'will by'}
def spacing_some_connect_words(text):
    """
    'Whyare' -> 'Why are'
    """
    for error in mis_spell_mapping:
        if error in text:
            text = text.replace(error, mis_spell_mapping[error])
            
    # what
    text = re.sub(r" (W|w)hat+(s)*[A|a]*(p)+ ", " WhatsApp ", text)
    text = re.sub(r" (W|w)hat\S ", " What ", text)
    text = re.sub(r" \S(W|w)hat ", " What ", text)
    # why
    text = re.sub(r" (W|w)hy\S ", " Why ", text)
    text = re.sub(r" \S(W|w)hy ", " Why ", text)
    # How
    text = re.sub(r" (H|h)ow\S ", " How ", text)
    text = re.sub(r" \S(H|h)ow ", " How ", text)
    # which
    text = re.sub(r" (W|w)hich\S ", " Which ", text)
    text = re.sub(r" \S(W|w)hich ", " Which ", text)
    # where
    text = re.sub(r" (W|w)here\S ", " Where ", text)
    text = re.sub(r" \S(W|w)here ", " Where ", text)
    # 
    text = mis_connect_re.sub(r" \1 ", text)
    text = text.replace("What sApp", 'WhatsApp')
    
    text = remove_space(text)
    return text

# clean repeated letters
def clean_repeat_words(text):
    text = text.replace("img", "ing")

    text = re.sub(r"(I|i)(I|i)+ng", "ing", text)
    text = re.sub(r"(L|l)(L|l)(L|l)+y", "lly", text)
    text = re.sub(r"(A|a)(A|a)(A|a)+", "a", text)
    text = re.sub(r"(C|c)(C|c)(C|c)+", "cc", text)
    text = re.sub(r"(D|d)(D|d)(D|d)+", "dd", text)
    text = re.sub(r"(E|e)(E|e)(E|e)+", "ee", text)
    text = re.sub(r"(F|f)(F|f)(F|f)+", "ff", text)
    text = re.sub(r"(G|g)(G|g)(G|g)+", "gg", text)
    text = re.sub(r"(I|i)(I|i)(I|i)+", "i", text)
    text = re.sub(r"(K|k)(K|k)(K|k)+", "k", text)
    text = re.sub(r"(L|l)(L|l)(L|l)+", "ll", text)
    text = re.sub(r"(M|m)(M|m)(M|m)+", "mm", text)
    text = re.sub(r"(N|n)(N|n)(N|n)+", "nn", text)
    text = re.sub(r"(O|o)(O|o)(O|o)+", "oo", text)
    text = re.sub(r"(P|p)(P|p)(P|p)+", "pp", text)
    text = re.sub(r"(Q|q)(Q|q)+", "q", text)
    text = re.sub(r"(R|r)(R|r)(R|r)+", "rr", text)
    text = re.sub(r"(S|s)(S|s)(S|s)+", "ss", text)
    text = re.sub(r"(T|t)(T|t)(T|t)+", "tt", text)
    text = re.sub(r"(V|v)(V|v)+", "v", text)
    text = re.sub(r"(Y|y)(Y|y)(Y|y)+", "y", text)
    text = re.sub(r"plzz+", "please", text)
    text = re.sub(r"(Z|z)(Z|z)(Z|z)+", "zz", text)
    return text

def preprocess(text):
    """
    preprocess text main steps
    """
    text = remove_space(text)
    text = pre_clean_unknown_words(text)
    text = clean_special_punctuations(text)
    text = clean_number(text)
    text = pre_clean_rare_words(text)
    text = decontracted(text)
    text = clean_latex(text)
    text = clean_misspell(text)
    text = spacing_punctuation(text)
    text = spacing_some_connect_words(text)
    text = clean_bad_case_words(text)
    text = clean_repeat_words(text)
    text = remove_space(text)
    return text


def text_clean_wrapper(df):
    df["question_text"] = df["question_text"].apply(preprocess)
    return df

def threshold_search(y_true, y_proba):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001) 
    F = 2/(1/precision + 1/recall)
    best_score = np.max(F)
    best_th = thresholds[np.argmax(F)]
    search_result = {'threshold': best_th, 'f1': best_score}
    return search_result

class GaussianNoise(nn.Module):
    def __init__(self, mean= 0., sigma=0.1):
        super().__init__()
        self._mean = mean
        self._sigma = sigma

    def forward(self, x):
        if self.training:
            return x + torch.autograd.Variable(torch.randn(x.size()).cuda() * self._sigma + self._mean)
        return x

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class GRUNet(nn.Module):
    def __init__(self, embedding_matrix):
        super(GRUNet, self).__init__()
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.gru_1 = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru_2 = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)
        
        #self.noise = GaussianNoise(sigma=0.1)
        
        self.gru_1_attention = Attention(hidden_size*2, maxlen)
        self.gru_2_attention = Attention(hidden_size*2, maxlen)
        
        self.linear = nn.Linear(hidden_size*8, 16)
        self.relu = nn.ReLU()

        self.out = nn.Linear(16, 1)
        
    def forward(self, x):
        
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        #h_embedding = self.noise(h_embedding)

        h_gru_1, _ = self.gru_1(h_embedding)
        h_gru_2, _ = self.gru_2(h_gru_1)
        
        h_gru_1_atten = self.gru_1_attention(h_gru_1)
        h_gru_2_atten = self.gru_2_attention(h_gru_2)
        
        avg_pool = torch.mean(h_gru_2, 1)
        max_pool, _ = torch.max(h_gru_2, 1)
        
        conc = torch.cat((h_gru_1_atten, h_gru_2_atten, avg_pool, max_pool), 1)
        
        conc = self.relu(self.linear(conc))
        out = self.out(conc)
        
        return out

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
# LOAD DATA
train_df = pd.read_csv(PATH+'train.csv', usecols=['question_text', 'target'])
test_df = pd.read_csv(PATH+'test.csv', usecols = ['question_text'])

train_df = df_parallelize_run(train_df, text_clean_wrapper)
test_df = df_parallelize_run(test_df, text_clean_wrapper)

# FOR CREATING PROCESSED DATA AND LABELS
train_sentences = train_df['question_text']
train_labels = train_df['target']
test_sentences = test_df['question_text']

del train_df, test_df

gc.collect()

# TOKENIZE TEXT
tokenizer = text.Tokenizer(num_words=max_features, oov_token='OOV')
tokenizer.fit_on_texts(list(train_sentences) + list(test_sentences))

tokenized_train = tokenizer.texts_to_sequences(train_sentences)
X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)

tokenized_test = tokenizer.texts_to_sequences(test_sentences)
X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)

del tokenized_test, tokenized_train, train_sentences, test_sentences
gc.collect()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index)+1)

# LIST OF ALL EMBEDDINGS USED
embedding_list = [PATH+'embeddings/paragram_300_sl999/paragram_300_sl999.txt', 
PATH+'embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',
PATH+'embeddings/glove.840B.300d/glove.840B.300d.txt']

# MEAN AND STD VALUES FOR EMBEDDINGS
emb_mean_dict = {'paragram_300_sl999':-0.005324783269315958,
            'wiki-news-300d-1M':-0.0033469984773546457,
            'glove.840B.300d':-0.005838498938828707}

emb_std_dict = {'paragram_300_sl999':0.4934646189212799,
            'wiki-news-300d-1M':0.10985549539327621,
            'glove.840B.300d':0.4878219664096832}

global_mean = np.mean([i for i in emb_mean_dict.values()])
global_std = np.mean([i for i in emb_std_dict.values()])
global_embedding = np.random.normal(global_mean, global_std, (nb_words, embed_size))
embedding_count = np.zeros((nb_words,1))

for EMBEDDING_FILE in embedding_list:
    embedding_name = EMBEDDING_FILE.split('/')[3]
    for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore'):
        word, vec = o.split(' ', 1)
        if word not in word_index:
            continue
        i = word_index[word]
        if i >= nb_words:
            continue
        embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:embed_size]
        if len(embedding_vector) == embed_size:
            if embedding_count[i] == 0:
                global_embedding[i] = embedding_vector
            else:
                global_embedding[i] = (embedding_count[i]*global_embedding[i] + embedding_vector)/(embedding_count[i] + 1)
            embedding_count[i] += 1
    del embedding_vector
    gc.collect()

set_seed(seed)

X_test = torch.tensor(X_test, dtype=torch.long).cuda()
test_tensor = torch.utils.data.TensorDataset(X_test)
test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

# TO SAVE FINAL PREDICTIONS
final_preds = np.zeros(len(X_test))

train_idx, valid_idx = list(StratifiedKFold(n_splits=50, shuffle=True, random_state=seed).split(X_train, train_labels))[0]

train_x = torch.tensor(X_train[train_idx], dtype=torch.long).cuda()
train_y = torch.tensor(train_labels[train_idx, np.newaxis], dtype=torch.float32).cuda()

valid_x = torch.tensor(X_train[valid_idx], dtype=torch.long).cuda()
valid_y = torch.tensor(train_labels[valid_idx, np.newaxis], dtype=torch.float32).cuda()

embedding_matrix = global_embedding.copy()

model = GRUNet(embedding_matrix)
model.cuda()

loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=lr_patience)

train_tensor = torch.utils.data.TensorDataset(train_x, train_y)
valid_tensor = torch.utils.data.TensorDataset(valid_x, valid_y)

train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_tensor, batch_size=batch_size, shuffle=False)

early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

test_preds_fold = np.zeros(len(X_test))
valid_preds_fold = np.zeros((valid_x.size(0)))

for epoch in range(epochs_fixed):
    start_time = time.time()
    model.train()
    avg_loss = 0.
    for x_batch, y_batch in tqdm(train_loader, disable=True):
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)
    
    model.eval()
    avg_val_loss = 0.
    for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_pred = model(x_batch).detach()
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
    
    elapsed_time = time.time() - start_time 
    print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
        epoch + 1, epochs_fixed, avg_loss, avg_val_loss, elapsed_time))
    scheduler.step(avg_val_loss)

    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

model.embedding.weight.requires_grad = True

model.load_state_dict(torch.load('checkpoint.pt'))
early_stopping.early_stop = False
early_stopping.counter = 0

for epoch in range(epochs_trainable):
    start_time = time.time()
    
    model.train()
    avg_loss = 0.
    for x_batch, y_batch in tqdm(train_loader, disable=True):
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)
    
    model.eval()
    avg_val_loss = 0.
    for x_batch, y_batch in valid_loader:
        y_pred = model(x_batch).detach()
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
    
    elapsed_time = time.time() - start_time 
    print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
        epoch + 1, epochs_trainable, avg_loss, avg_val_loss, elapsed_time))
    scheduler.step(avg_val_loss)

    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

model.load_state_dict(torch.load('checkpoint.pt'))

for i, (x_batch,) in enumerate(test_loader):
    y_pred = model(x_batch).detach()

    test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

model.eval()
avg_val_loss = 0.
for i, (x_batch, y_batch) in enumerate(valid_loader):
    y_pred = model(x_batch).detach()
    avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
    valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

print('Making oof preds with model with valid loss: {}\n'.format(avg_val_loss))
final_preds = test_preds_fold 

gc.collect()

optimal_threshold = threshold_search(valid_y.cpu().numpy(), valid_preds_fold)
print(optimal_threshold)

with open('results_log.txt', 'a') as f:
    f.write(change_string)
    f.write(str(time.localtime()))
    f.write('\n')
    f.write(str(optimal_threshold))
    f.write('\n\n')

# SUBMISSION FILE
final_preds = (final_preds > optimal_threshold['threshold']).astype(int)
sample = pd.read_csv(PATH+'sample_submission.csv')
sample['prediction'] = final_preds
sample.to_csv('submission.csv', index=False)