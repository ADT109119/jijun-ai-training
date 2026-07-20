import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trainer.validate_json import extract_tool_call, validate_record

CATEGORIES = [
    "餐飲飲食", "休閒娛樂", "交通出行", "生活繳費", "醫療保健",
    "教育學習", "日常雜貨", "服飾美妝", "數位服務", "投資理財",
    "薪資收入", "獎金紅利", "副業外快", "寵物支出", "房租房貸",
    "保險費用", "人情往來", "家居裝修", "3C電子", "運動健身"
]

ACCOUNTS = [
    "現金", "信用卡", "悠遊卡", "一卡通", "街口支付",
    "LINE Pay", "Apple Pay", "Google Pay", "郵局帳戶",
    "銀行存款", "外幣帳戶", "加密貨幣", "悠遊付", "icash"
]

# ─── Level-2 Noise templates ───
NOISE_TEXTS_EXPENSE = [
    # 餐飲
    lambda c,a: f"今天中午跟同事去{a}附近新開的那家{c}吃了商業午餐，味道還不錯但價格有點小貴總共花了{a.split()[0] if len(a.split())>1 else ''}好飽喔～",
    lambda c,a: f"下班後超餓的，路過{c}買了個便當跟飲料，老闆人很好還多送了一碗湯，總共{a}元用{a}結帳的😋",
    lambda c,a: f"哇今天跟好久不見的朋友約吃{c}，聊了三個小時超開心，吃下來一個人{a}元刷信用卡，值得啦！",
    lambda c,a: f"昨天叫了熊貓外送點{c}，滿額折價後只要{a}元用LINE Pay付款，下雨天懶得出門的好選擇～",
    lambda c,a: f"早上趕時間去{c}買了三明治跟大冰拿，花{a}元用悠遊卡嗶一下就走超方便！",
    # 休閒娛樂
    lambda c,a: f"週末去{c}玩了一整天，門票加餐飲總共花了{a}元，用信用卡買票還有打折耶，開心！",
    lambda c,a: f"昨天跟朋友去唱KTV從下午唱到晚上，一個人分攤{a}元用{a}結帳，唱到燒聲了哈哈😂",
    lambda c,a: f"Netflix這個月又漲價了啦，但還是繼續訂因為太多劇想追了，月費{a}元刷信用卡，宅宅日常～",
    lambda c,a: f"今天去逛{c}本來只是隨便走走，結果不小心手滑買了東西花了{a}元用LINE Pay⋯錢包對不起🥲",
    # 交通
    lambda c,a: f"今天上下班都搭{a}，早上刷了一次下午又刷了一次，一天交通費{a}元，比騎車還省油錢啦！",
    lambda c,a: f"下雨天不想淋雨叫了Uber去公司，結果塞車花了{a}元用信用卡付款，心痛比雨還大😭",
    lambda c,a: f"上週{a}儲值了{a}元，結果這一週每天搭捷運公車用到現在快見底了，台北通勤真的好花錢啊～",
    lambda c,a: f"今天騎{a}去辦事，還車的時候扣了{a}元，半小時內免費的政策沒了以後好貴⋯",
    # 日常雜貨
    lambda c,a: f"去{c}補貨買了衛生紙洗碗精跟零食，結帳{a}元用{a}付的，每次去{c}沒有千元走不出來😂",
    lambda c,a: f"全聯週六有會員日優惠！買了牛奶雞蛋跟一些蔬果花了{a}元用{a}結帳，省了大概50塊吧～",
    lambda c,a: f"家裡洗髮精沐浴乳都用完了，去屈臣氏補了一批花了{a}元刷信用卡，還好有活動買一送一💪",
    lambda c,a: f"今天去好市多補貨，一大車東西結帳{a}元用{a}付款，每次去都覺得自己是土豪結完帳就後悔了🤣",
    # 服飾美妝
    lambda c,a: f"換季了去{c}買了幾件衣服，特價區挖到寶總共{a}元用{a}刷的，衣櫃又爆炸了但好開心🎉",
    lambda c,a: f"週年慶真的太可怕了⋯買了一組保養品跟兩支口紅花了{a}元刷信用卡，但贈品拿得好爽😂",
    lambda c,a: f"路過{c}看到櫥窗那件外套太好看了，試穿後直接買了{a}元用Apple Pay，衝動購物的我沒救了～",
    # 3C/數位
    lambda c,a: f"買了新的{a}保護殼加充電線總共{a}元用{a}付款，舊的用了兩年終於退役了～",
    lambda c,a: f"Steam秋季特賣又來了！買了好幾款願望清單的遊戲總共{a}元刷信用卡，錢包已死有事燒紙💀",
    lambda c,a: f"Spotify家庭方案這個月由我主揪，六個人分攤一個人{a}元，六個都收齊了用LINE Pay轉給我～",
    # 寵物
    lambda c,a: f"家裡那隻挑嘴貓只有某牌罐頭才吃，今天去補貨買了兩打花{a}元用{a}結帳，主子開心就好😺",
    lambda c,a: f"帶狗狗去洗澡加剪毛花了{a}元用{a}付款，洗完變超帥的狗界歐巴～🐕",
    # 醫療
    lambda c,a: f"最近過敏性鼻炎又發作了去藥局買噴劑跟藥花了{a}元用{a}付，鼻子暢通的感覺真好～",
    lambda c,a: f"今天去复健科做物理治療，掛號加自費療程{a}元刷信用卡，長期坐辦公室的職業傷害啊⋯",
    # 教育
    lambda c,a: f"買了{c}的線上課程特價只要{a}元用{a}付款，趁打折入手充實一下自己💪",
    lambda c,a: f"去書局買了兩本{c}相關的書跟一本筆記本總共{a}元用{a}結帳，好久沒認真看書了要加油📚",
    # 家居
    lambda c,a: f"去IKEA買了個收納櫃回家自己組，花了{a}元用{a}付款，DIY的樂趣無窮但手好痠😅",
    lambda c,a: f"房間缺一盞檯燈去生活工場買了一盞設計款{a}元用{a}結帳，房間氣氛瞬間升級了～",
    # 運動
    lambda c,a: f"報名下個月的路跑活動報名費{a}元用{a}付款，為了這個要開始訓練了不然跑不完🏃",
    lambda c,a: f"買了一張瑜伽墊跟兩顆彈力帶總共{a}元用{a}結帳，在家運動省健身房月費也不錯～",
    # 生活繳費
    lambda c,a: f"收到這期的{a}帳單{a}元，順手用{a}繳掉了，這種固定支出每個月都好幾筆😮‍💨",
    lambda c,a: f"手機帳單來了{a}元用{a}自動扣繳，4G吃到飽用習慣了懶得換～",
    # 人情
    lambda c,a: f"朋友生日請她吃了一頓{c}花了{a}元用{a}結帳，生日快樂呀～最好的朋友值得🎂",
    lambda c,a: f"同事結婚大家一起合買禮物我出了{a}元用{a}轉給主揪，希望他們幸福久久💑",
    # 保險
    lambda c,a: f"{c}保費又扣款了{a}元從{a}自動轉帳，雖然每個月多一筆但買個安心啦～",
    lambda c,a: f"幫毛小孩保的寵物險月繳{a}元從{a}自動扣，狗狗也是家人要好好保護牠🐾",
    # 投資
    lambda c,a: f"每月定期定額{a}扣款{a}元，不知不覺也存了好幾年了持續累積被動收入📈",
]
NOISE_TEXTS_INCOME = [
    lambda c,a: f"耶！今天發薪日！{c}進來了{a}元入{a}，這個月終於不用吃土了🎉",
    lambda c,a: f"年終獎金進來了！{c}{a}元匯到{a}，開心到飛起來～過年可以包大包一點了🧧",
    lambda c,a: f"接了一個外包案子今天收到{c}，{a}元直接入{a}，副業收入越來越穩定了💪",
    lambda c,a: f"股票配息入帳啦！持有{a}的{c}{a}元進{a}帳戶，被動收入讚讚的📈",
    lambda c,a: f"今天收到{c}退稅{a}元直接匯到{a}，不無小補每年這時候都小確幸～",
    lambda c,a: f"房租收入進來了！這個月租客按時匯款{a}元到{a}，穩定的被動收入真棒🏠",
    lambda c,a: f"賣掉二手手機跟一些用不到的3C用品，總共賣了{a}元匯到{a}，斷捨離還有錢賺太讚了！",
    lambda c,a: f"公司發的績效獎金{a}元入{a}了，上半年的努力沒有白費🥹",
    lambda c,a: f"幫朋友接了一個翻譯案子完成後收到{c}，{a}元入{a}，語言能力真的可以變現耶～",
    lambda c,a: f"美金定存到期了利息加本金總共{a}元入外幣帳戶，被動收入持續累積中🌏",
]

# ─── Level-3 Reasoning templates ───
REASON_TEXTS_EXPENSE = [
    lambda c,a: f"這週五天上班日午餐平均一天{a}元，但週三跟客戶吃飯是公司招待不算，所以四天午餐總共{a}元用{a}付的幫我算一下記帳",
    lambda c,a: f"跟三個朋友去吃{c}總帳單{a}元我先刷卡付了，他們說要各轉{a}元給我，幫我記我實際負擔的部分就好",
    lambda c,a: f"這個月去了三次{a}，第一次{a}元第二次{a}元第三次{a}元總共刷同一張信用卡，幫我加總記一筆",
    lambda c,a: f"昨天買衣服{a}元，今天又去買鞋子{a}元，都是同家店刷卡消費，幫我記成同一天的支出",
    lambda c,a: f"上週跟這週各去了一次好市多，上週花{a}元這週花了{a}元都用同一張卡，幫我算這兩次總共多少錢記下來",
    lambda c,a: f"今天領包裹的時候順便買了{a}{a}元，但這個是幫同事代買的他明天會還我錢，所以先幫我記但備註寫清楚",
    lambda c,a: f"我每天搭公車上下班一趟{a}元來回{a}元，一個月上班22天，幫我算一個月交通費總共多少記下來",
    lambda c,a: f"昨天跟朋友去吃{c}總帳單{a}元總共五個人但壽星免費所以四個人分攤，幫我算一個人多少錢記帳",
    lambda c,a: f"這個月收到電費帳單{a}元、水費{a}元、瓦斯{a}元，三筆都從銀行帳戶扣了，幫我加總記一筆",
    lambda c,a: f"報名了一個線上課程{a}元加教材費{a}元總共{a}元用{a}付款，幫我記整筆支出",
    lambda c,a: f"今天去健身房繳了入會費{a}元加第一個月月費{a}元總共{a}元用{a}結帳，幫我合計一筆",
    lambda c,a: f"訂了兩箱貓砂{a}元跟三箱罐頭{a}元總共{a}元用{a}付款，幫我合併記一筆寵物支出",
    lambda c,a: f"買了專業級{a}{a}元還買了保護套{a}元總共{a}元用{a}，幫我記一筆",
    lambda c,a: f"今天去醫院掛了兩科，眼科{a}元牙科{a}元總共{a}元刷信用卡，幫我合計一筆醫療保健支出",
    lambda c,a: f"這個月訂閱了Netflix{a}元Spotify{a}元跟iCloud{a}元總共{a}元都刷同一張信用卡，幫我記一筆數位服務總支出",
]

REASON_TEXTS_INCOME = [
    lambda c,a: f"本薪{a}元加伙食津貼{a}元總共{a}元入{a}，但這是這個月的總收入幫我分開記也可以合計",
    lambda c,a: f"上週接了一個案子今天收到訂金{a}元說下週完工再付尾款{a}元總共{a}元，先記今天這筆就好",
    lambda c,a: f"公司發了中秋禮金{a}元跟端午禮金{a}元一起入了{a}帳戶，這是兩個節日的幫我各記一筆或記總額",
    lambda c,a: f"股票配息{a}元加上基金配息{a}元總共{a}元入{a}帳戶，被動收入幫我一筆記",
    lambda c,a: f"賣掉一些用不到的3C產品總共賣了{a}元扣掉平台手續費{a}元實拿{a}元入{a}，幫我記實收金額",
]

def build_sample(diff_level, cat, acc, amt, desc, rtype, user_text):
    sub_cats = random.sample(CATEGORIES, k=random.randint(4, 6))
    if cat not in sub_cats:
        sub_cats[-1] = cat
    sub_accs = random.sample(ACCOUNTS, k=random.randint(3, 5))
    if acc not in sub_accs:
        sub_accs[-1] = acc
    tool_def = {
        "name": "add_record",
        "description": "新增一筆記帳記錄",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "category": {"type": "string", "enum": sub_cats},
                "account": {"type": "string", "enum": sub_accs},
                "description": {"type": "string"},
                "type": {"type": "string", "enum": ["expense", "income"]}
            },
            "required": ["amount", "category", "account", "type"]
        }
    }
    system_content = "你是一個記帳助理。你被賦予了以下 tools:\n" + json.dumps(tool_def, ensure_ascii=False)
    args_dict = {"amount": amt, "category": cat, "account": acc, "description": desc, "type": rtype}
    inner = json.dumps({"name": "add_record", "args": args_dict}, ensure_ascii=True)
    assistant_content = "<tool_call>" + inner + "</tool_call>"
    return {
        "difficulty_level": diff_level,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def generate_batch(count_l1=0, count_l2=0, count_l3=0):
    samples = []
    used_signatures = set()

    def gen_one(diff, cat, acc, amt, desc, rtype, text_fn):
        # Build text with proper account/amount substitution
        try:
            text = text_fn(cat, acc) if "{cat}" in text_fn.__code__.co_varnames[:text_fn.__code__.co_argcount] else text_fn(cat, acc)
        except:
            text = text_fn(cat, acc)
        sig = (diff, cat, acc, amt, desc, rtype, text[:30])
        if sig in used_signatures:
            return None
        used_signatures.add(sig)
        sample = build_sample(diff, cat, acc, amt, desc, rtype, text)
        try:
            ast = sample["messages"][2]["content"]
            args = extract_tool_call(ast)
            valid, err = validate_record(args)
            if not valid:
                return None
        except:
            return None
        return sample

    # Level-2 Noise generation
    for _ in range(count_l2):
        cat = random.choice(CATEGORIES)
        acc = random.choice(ACCOUNTS)
        rtype = "expense" if random.random() < 0.7 else "income"
        if rtype == "expense":
            amt = random.choice([75, 120, 150, 180, 200, 250, 280, 320, 350, 400, 450, 500, 550, 600, 650, 700, 780, 800, 850, 900, 950, 990, 1000, 1200, 1280, 1350, 1500, 1680, 1800, 2000, 2200, 2500, 2800, 3000, 3500, 3990, 4500, 5000, 6800, 8000, 10000, 12000, 15000, 18000, 20000])
            desc_pool = [
                f"在{cat}的消費", f"購買{cat}相關商品", f"{cat}{acc}付款",
                f"{random.choice(['日常','週末','下班後','上班'])}{cat}消費",
                f"{random.choice(['和朋友','和家人','自己'])}去{cat}"
            ]
            if random.random() < 0.05:
                rtype = "income"
        else:
            amt = random.choice([1500, 2000, 2500, 2800, 3000, 3500, 4000, 4500, 5000, 6000, 8000, 10000, 12000, 15000, 18000, 20000, 25000, 28000, 30000, 35000, 38000, 40000, 42000, 45000, 48000, 50000, 52000, 55000, 60000, 80000, 100000])
            desc_pool = [
                f"{random.choice(['月薪','兼職','打工'])}收入",
                f"{random.choice(['接案','外快','副業'])}報酬",
                f"{random.choice(['年終','三節','績效','分紅'])}獎金",
                f"{random.choice(['股息','配息','利息','投資'])}收益",
                f"{random.choice(['退稅','補助','津貼'])}入帳"
            ]
        desc = random.choice(desc_pool)
        text_fn = random.choice(NOISE_TEXTS_INCOME if rtype == "income" else NOISE_TEXTS_EXPENSE)
        sample = gen_one("Level-2 (Noise)", cat, acc, amt, desc, rtype, text_fn)
        if sample:
            samples.append(sample)

    # Level-3 Reasoning generation
    for _ in range(count_l3):
        cat = random.choice(CATEGORIES)
        acc = random.choice(ACCOUNTS)
        rtype = "expense" if random.random() < 0.65 else "income"
        if rtype == "expense":
            amt = random.choice([150, 250, 360, 480, 540, 640, 720, 850, 960, 1080, 1280, 1500, 1800, 2100, 2500, 3000, 3500, 4000, 4800, 5600, 6500, 7500, 9000, 10000, 12000, 15000, 18500, 20000, 25000, 30000])
            desc_pool = [
                f"{random.choice(['合計','加總','分攤'])}{cat}費用",
                f"{random.choice(['多次','多筆','合併'])}{cat}支出",
                f"{random.choice(['計算後','推算','結算'])}{cat}{random.choice(['花費','支出','帳單'])}"
            ]
        else:
            amt = random.choice([1200, 2500, 3200, 4500, 5000, 6000, 7500, 8000, 9000, 10000, 12000, 15000, 18000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000])
            desc_pool = [
                f"{random.choice(['合計','多筆'])}{random.choice(['收入','報酬','收益'])}",
                f"{random.choice(['結算','匯整'])}{cat}款項"
            ]
        desc = random.choice(desc_pool)
        text_fn = random.choice(REASON_TEXTS_INCOME if rtype == "income" else REASON_TEXTS_EXPENSE)
        sample = gen_one("Level-3 (Reasoning)", cat, acc, amt, desc, rtype, text_fn)
        if sample:
            samples.append(sample)

    # Level-1 Simple generation
    l1_expense_templates = [
        ("買{c}用{a}", ["去","在"], ["店裡","超商","賣場"], ["飲料","零食","麵包"], "花了{amt}元用{a}付的。"),
        ("{cat}消費{a}", [], [], [], "剛剛在{c}花了{amt}元用{a}付款。"),
        ("日常{cat}", ["",], ["",], ["",], "{amt}元用{a}付的，買了{c}。"),
        ("週末{cat}", ["",], ["",], ["",], "{amt}元用{a}結帳，買了{c}。"),
        ("{cat}{a}", [], [], [], "今天去{c}花了{amt}元，用{a}結帳。"),
    ]
    l1_income_templates = [
        "{cat}入帳",
        "{cat}",
    ]
    l1_income_texts = [
        "收到{cat}{amt}元匯到{a}了。",
        "{cat}{amt}元已經入{a}帳戶。",
        "{cat}{amt}元入帳到{a}。",
        "今天{a}收到了{cat}{amt}元。",
    ]
    for _ in range(count_l1):
        rtype = "expense" if random.random() < 0.6 else "income"
        cat = random.choice(CATEGORIES)
        acc = random.choice(ACCOUNTS)
        if rtype == "expense":
            amt = random.choice([30, 45, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 250, 280, 300, 320, 350, 380, 400, 420, 450, 480, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1280, 1350, 1500, 1600, 1800, 2000, 2200, 2500, 2800, 3000, 3500, 3990, 4200, 4500, 5000, 5500, 6000, 6800, 7000, 8000, 9000, 10000, 12000, 12800, 15000, 18000, 20000, 25000, 30000])
            desc_tmpl, place_tmpl, store_tmpl, item_tmpl, text_tmpl = random.choice(l1_expense_templates)
            place = random.choice(place_tmpl) if place_tmpl else ""
            store = random.choice(store_tmpl) if store_tmpl else ""
            item = random.choice(item_tmpl) if item_tmpl else ""
            user_text = text_tmpl.format(c=cat, a=acc, amt=amt, place=place, store=store, item=item)
            desc = desc_tmpl.format(cat=cat, a=acc, amt=amt, place=place, store=store, item=item)
        else:
            amt = random.choice([1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 22000, 25000, 28000, 30000, 32000, 35000, 38000, 40000, 42000, 45000, 48000, 50000, 52000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 100000])
            desc_tmpl = random.choice(l1_income_templates)
            text_tmpl = random.choice(l1_income_texts)
            user_text = text_tmpl.format(cat=cat, a=acc, amt=amt)
            desc = desc_tmpl.format(cat=cat, a=acc, amt=amt)
        sample = gen_one("Level-1 (Simple)", cat, acc, amt, desc, rtype, lambda c, a: user_text)
        if sample:
            samples.append(sample)

    return samples

def main():
    raw_path = "dataset/raw_generated.jsonl"

    # Current stats
    current = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                current.append(json.loads(line))
    print(f"Current: {len(current)} samples")

    # Validate all existing
    print("Validating existing...")
    fail = 0
    for s in current:
        ast = [m["content"] for m in s["messages"] if m["role"] == "assistant"][0]
        try:
            args = extract_tool_call(ast)
            valid, _ = validate_record(args)
            if not valid:
                fail += 1
        except:
            fail += 1
    print(f"Existing: {len(current)} passed, {fail} failed")

    need = 1000 - len(current)
    print(f"Need {need} more samples")

    # Target distribution: ~35% L1, ~35% L2, ~30% L3 (more complex = better)
    target_l1 = int(need * 0.30)
    target_l2 = int(need * 0.35)
    target_l3 = need - target_l1 - target_l2

    batch_size = 200
    all_new = []

    for batch_idx in range(0, need, batch_size):
        b_l1 = min(target_l1 // (need // batch_size + 1), batch_size // 3) if batch_idx == 0 else 0
        b_l2 = min(batch_size, need - len(all_new)) // 2
        b_l3 = batch_size - b_l1 - b_l2

        # Actually let's just split evenly per batch
        remaining = need - len(all_new)
        batch_now = min(batch_size, remaining)
        b_l1 = int(batch_now * 0.30)
        b_l2 = int(batch_now * 0.35)
        b_l3 = batch_now - b_l1 - b_l2

        print(f"\nBatch {batch_idx // batch_size + 1}: generating {batch_now} samples (L1:{b_l1}, L2:{b_l2}, L3:{b_l3})...")
        new_samples = generate_batch(count_l1=b_l1, count_l2=b_l2, count_l3=b_l3)
        all_new.extend(new_samples)
        print(f"  Got {len(new_samples)} valid samples (total new: {len(all_new)})")

        # Write incrementally
        with open(raw_path, "a", encoding="utf-8") as f:
            for s in new_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        if len(all_new) >= need:
            break

    print(f"\nGenerated total: {len(all_new)} new samples")

    # Final validation
    print("\nFinal validation of all samples...")
    all_samples = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line))

    counts = {"Level-1 (Simple)": 0, "Level-2 (Noise)": 0, "Level-3 (Reasoning)": 0}
    inc = 0
    exp = 0
    passed = 0
    failed = 0
    for s in all_samples:
        diff = s.get("difficulty_level", "")
        if "Level-1" in diff:
            counts["Level-1 (Simple)"] += 1
        elif "Level-2" in diff:
            counts["Level-2 (Noise)"] += 1
        elif "Level-3" in diff:
            counts["Level-3 (Reasoning)"] += 1
        ast = [m["content"] for m in s["messages"] if m["role"] == "assistant"][0]
        if "income" in ast:
            inc += 1
        else:
            exp += 1
        try:
            args = extract_tool_call(ast)
            valid, _ = validate_record(args)
            if valid:
                passed += 1
            else:
                failed += 1
        except:
            failed += 1

    print(f"Total: {len(all_samples)}")
    print(f"L1: {counts['Level-1 (Simple)']}, L2: {counts['Level-2 (Noise)']}, L3: {counts['Level-3 (Reasoning)']}")
    print(f"Income: {inc}, Expense: {exp}")
    print(f"Parser: {passed} passed, {failed} failed")

    # Regenerate splits
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    train = all_samples[:split_idx]
    test = all_samples[split_idx:]
    with open("dataset/train_strata.jsonl", "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open("dataset/test_strata.jsonl", "w", encoding="utf-8") as f:
        for s in test:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\ntrain_strata.jsonl: {len(train)} samples")
    print(f"test_strata.jsonl: {len(test)} samples")

if __name__ == "__main__":
    main()
