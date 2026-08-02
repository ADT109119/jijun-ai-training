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
# Each lambda takes (c=category, a=account, m=amount)
NOISE_TEXTS_EXPENSE = [
    lambda c,a,m: f"今天中午跟同事去公司附近新開的那家{c}吃了商業午餐，味道還不錯但價格有點小貴總共花了{m}元好飽喔～",
    lambda c,a,m: f"下班後超餓的，路過{c}買了個便當跟飲料，老闆人很好還多送了一碗湯，總共{m}元用{a}結帳的😋",
    lambda c,a,m: f"哇今天跟好久不見的朋友約吃{c}，聊了三個小時超開心，吃下來一個人{m}元刷信用卡，值得啦！",
    lambda c,a,m: f"昨天叫了熊貓外送點{c}，滿額折價後只要{m}元用LINE Pay付款，下雨天懶得出門的好選擇～",
    lambda c,a,m: f"早上趕時間去{c}買了三明治跟大冰拿，花{m}元用悠遊卡嗶一下就走超方便！",
    lambda c,a,m: f"週末去{c}玩了一整天，門票加餐飲總共花了{m}元，用信用卡買票還有打折耶，開心！",
    lambda c,a,m: f"昨天跟朋友去唱KTV從下午唱到晚上，一個人分攤{m}元用{a}結帳，唱到燒聲了哈哈😂",
    lambda c,a,m: f"Netflix這個月又漲價了啦，但還是繼續訂因為太多劇想追了，月費{m}元刷信用卡，宅宅日常～",
    lambda c,a,m: f"今天去逛{c}本來只是隨便走走，結果不小心手滑買了東西花了{m}元用LINE Pay⋯錢包對不起🥲",
    lambda c,a,m: f"今天上下班都搭{a}，早上刷了一次下午又刷了一次，一天交通費{m}元，比騎車還省油錢啦！",
    lambda c,a,m: f"下雨天不想淋雨叫了Uber去公司，結果塞車花了{m}元用信用卡付款，心痛比雨還大😭",
    lambda c,a,m: f"上週{a}儲值了{m}元，結果這一週每天搭捷運公車用到現在快見底了，台北通勤真的好花錢啊～",
    lambda c,a,m: f"今天騎{a}去辦事，還車的時候扣了{m}元，半小時內免費的政策沒了以後好貴⋯",
    lambda c,a,m: f"去{c}補貨買了衛生紙洗碗精跟零食，結帳{m}元用{a}付的，每次去{c}沒有千元走不出來😂",
    lambda c,a,m: f"全聯週六有會員日優惠！買了牛奶雞蛋跟一些蔬果花了{m}元用{a}結帳，省了大概50塊吧～",
    lambda c,a,m: f"家裡洗髮精沐浴乳都用完了，去屈臣氏補了一批花了{m}元刷信用卡，還好有活動買一送一💪",
    lambda c,a,m: f"今天去好市多補貨，一大車東西結帳{m}元用{a}付款，每次去都覺得自己是土豪結完帳就後悔了🤣",
    lambda c,a,m: f"換季了去{c}買了幾件衣服，特價區挖到寶總共{m}元用{a}刷的，衣櫃又爆炸了但好開心🎉",
    lambda c,a,m: f"週年慶真的太可怕了⋯買了一組保養品跟兩支口紅花了{m}元刷信用卡，但贈品拿得好爽😂",
    lambda c,a,m: f"路過{c}看到櫥窗那件外套太好看了，試穿後直接買了{m}元用Apple Pay，衝動購物的我沒救了～",
    lambda c,a,m: f"買了新的3C保護殼加充電線總共{m}元用{a}付款，舊的用了兩年終於退役了～",
    lambda c,a,m: f"Steam秋季特賣又來了！買了好幾款願望清單的遊戲總共{m}元刷信用卡，錢包已死有事燒紙💀",
    lambda c,a,m: f"Spotify家庭方案這個月由我主揪，六個人分攤一個人{m}元，六個都收齊了用LINE Pay轉給我～",
    lambda c,a,m: f"家裡那隻挑嘴貓只有某牌罐頭才吃，今天去補貨買了兩打花{m}元用{a}結帳，主子開心就好😺",
    lambda c,a,m: f"帶狗狗去洗澡加剪毛花了{m}元用{a}付款，洗完變超帥的狗界歐巴～🐕",
    lambda c,a,m: f"最近過敏性鼻炎又發作了去藥局買噴劑跟藥花了{m}元用{a}付，鼻子暢通的感覺真好～",
    lambda c,a,m: f"今天去复健科做物理治療，掛號加自費療程{m}元刷信用卡，長期坐辦公室的職業傷害啊⋯",
    lambda c,a,m: f"買了{c}的線上課程特價只要{m}元用{a}付款，趁打折入手充實一下自己💪",
    lambda c,a,m: f"去書局買了兩本{c}相關的書跟一本筆記本總共{m}元用{a}結帳，好久沒認真看書了要加油📚",
    lambda c,a,m: f"去IKEA買了個收納櫃回家自己組，花了{m}元用{a}付款，DIY的樂趣無窮但手好痠😅",
    lambda c,a,m: f"房間缺一盞檯燈去生活工場買了一盞設計款{m}元用{a}結帳，房間氣氛瞬間升級了～",
    lambda c,a,m: f"報名下個月的路跑活動報名費{m}元用{a}付款，為了這個要開始訓練了不然跑不完🏃",
    lambda c,a,m: f"買了一張瑜伽墊跟兩顆彈力帶總共{m}元用{a}結帳，在家運動省健身房月費也不錯～",
    lambda c,a,m: f"收到這期的{c}帳單{m}元，順手用{a}繳掉了，這種固定支出每個月都好幾筆😮‍💨",
    lambda c,a,m: f"手機帳單來了{m}元用{a}自動扣繳，4G吃到飽用習慣了懶得換～",
    lambda c,a,m: f"朋友生日請她吃了一頓{c}花了{m}元用{a}結帳，生日快樂呀～最好的朋友值得🎂",
    lambda c,a,m: f"同事結婚大家一起合買禮物我出了{m}元用{a}轉給主揪，希望他們幸福久久💑",
    lambda c,a,m: f"{c}保費又扣款了{m}元從{a}自動轉帳，雖然每個月多一筆但買個安心啦～",
    lambda c,a,m: f"幫毛小孩保的寵物險月繳{m}元從{a}自動扣，狗狗也是家人要好好保護牠🐾",
    lambda c,a,m: f"每月定期定額{a}扣款{m}元，不知不覺也存了好幾年了持續累積被動收入📈",
]
NOISE_TEXTS_INCOME = [
    lambda c,a,m: f"耶！今天發薪日！{c}進來了{m}元入{a}，這個月終於不用吃土了🎉",
    lambda c,a,m: f"年終獎金進來了！{c}{m}元匯到{a}，開心到飛起來～過年可以包大包一點了🧧",
    lambda c,a,m: f"接了一個外包案子今天收到{c}，{m}元直接入{a}，副業收入越來越穩定了💪",
    lambda c,a,m: f"股票配息入帳啦！持有股票的{c}{m}元進{a}帳戶，被動收入讚讚的📈",
    lambda c,a,m: f"今天收到{c}退稅{m}元直接匯到{a}，不無小補每年這時候都小確幸～",
    lambda c,a,m: f"房租收入進來了！這個月租客按時匯款{m}元到{a}，穩定的被動收入真棒🏠",
    lambda c,a,m: f"賣掉二手手機跟一些用不到的3C用品，總共賣了{m}元匯到{a}，斷捨離還有錢賺太讚了！",
    lambda c,a,m: f"公司發的績效獎金{m}元入{a}了，上半年的努力沒有白費🥹",
    lambda c,a,m: f"幫朋友接了一個翻譯案子完成後收到{c}，{m}元入{a}，語言能力真的可以變現耶～",
    lambda c,a,m: f"美金定存到期了利息加本金總共{m}元入外幣帳戶，被動收入持續累積中🌏",
]

# ─── Level-3 Reasoning templates ───
# Each lambda takes (c=category, a=account, m=amount)
# The amounts in the text must be consistent with the label amount m
REASON_TEXTS_EXPENSE = [
    lambda c,a,m: f"這週五天上班日午餐平均一天{m}元，但週三跟客戶吃飯是公司招待不算，所以四天午餐總共{m}元用{a}付的幫我算一下記帳",
    lambda c,a,m: f"跟三個朋友去吃{c}總帳單{m}元我先刷卡付了，他們說要各轉{int(m/4)}元給我，幫我記我實際負擔的部分就好",
    lambda c,a,m: f"這個月去了三次{c}，每次大概{m}元總共刷同一張{a}，幫我加總記一筆",
    lambda c,a,m: f"昨天買衣服{int(m/2)}元，今天又去買鞋子{int(m/2)}元，都是同家店刷卡消費用{a}，幫我記成同一天的支出",
    lambda c,a,m: f"上週跟這週各去了一次，上週花{int(m/2)}元這週花了{int(m/2)}元都用同一張{a}，幫我算這兩次總共多少錢記下來",
    lambda c,a,m: f"今天領包裹的時候順便買了東西花了{m}元用{a}，但這個是幫同事代買的他明天會還我錢，所以先幫我記但備註寫清楚",
    lambda c,a,m: f"我每天搭公車上下班一趟{int(round(m/44))}元來回{int(round(m/22))}元，一個月上班22天，幫我算一個月交通費總共多少記下來",
    lambda c,a,m: f"昨天跟朋友去吃{c}總帳單{m}元總共五個人但壽星免費所以四個人分攤，幫我算一個人多少錢記帳",
    lambda c,a,m: f"這個月收到電費帳單{int(m/3)}元、水費{int(m/3)}元、瓦斯{int(m/3)}元，三筆都從{a}扣了，幫我加總記一筆",
    lambda c,a,m: f"報名了一個線上課程{m}元用{a}付款，幫我記整筆支出",
    lambda c,a,m: f"今天去健身房繳了入會費加月費總共{m}元用{a}結帳，幫我合計一筆",
    lambda c,a,m: f"訂了寵物用品總共{m}元用{a}付款，幫我合併記一筆寵物支出",
    lambda c,a,m: f"買了設備加保護套總共{m}元用{a}，幫我記一筆",
    lambda c,a,m: f"今天去醫院掛了兩科總共{m}元刷{a}，幫我合計一筆醫療保健支出",
    lambda c,a,m: f"這個月訂閱了幾個服務總共{m}元都刷同一張{a}，幫我記一筆數位服務總支出",
]

REASON_TEXTS_INCOME = [
    lambda c,a,m: f"本薪加各項津貼總共{m}元入{a}，但這是這個月的總收入幫我分開記也可以合計",
    lambda c,a,m: f"上週接了一個案子今天收到訂金{int(m*0.3)}元說下週完工再付尾款{int(m*0.7)}元總共{m}元，先記今天這筆就好",
    lambda c,a,m: f"公司發了獎金總共{m}元入了{a}帳戶，這是一整筆的幫我記",
    lambda c,a,m: f"股票配息加上基金配息總共{m}元入{a}帳戶，被動收入幫我一筆記",
    lambda c,a,m: f"賣掉一些用不到的3C產品總共賣了{int(m*0.9)}元扣掉平台手續費{int(m*0.1)}元實拿{m}元入{a}，幫我記實收金額",
]

def build_sample(diff_level, cat, acc, amt, desc, rtype, user_text):
    today = "2026-07-20"
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
                "type": {"type": "string", "enum": ["expense", "income"]},
                "date": {"type": "string", "description": "ISO 8601 格式日期，例如 YYYY-MM-DD"}
            },
            "required": ["amount", "category", "account", "type", "date"]
        }
    }
    system_content = f"今天是 {today}。你是一個記帳助理。你被賦予了以下 tools:\n" + json.dumps(tool_def, ensure_ascii=False)
    args_dict = {"amount": amt, "category": cat, "account": acc, "description": desc, "type": rtype, "date": today}
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
        try:
            text = text_fn(cat, acc, amt)
        except TypeError:
            try:
                text = text_fn(cat, acc)
            except:
                text = text_fn(cat, acc)
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
        ("{c}消費", [], [], [], "剛剛在{c}花了{amt}元用{a}付款。"),
        ("日常{c}", [], [], [], "{amt}元用{a}付的，買了{c}。"),
        ("週末{c}", [], [], [], "{amt}元用{a}結帳，買了{c}。"),
        ("{c}{a}", [], [], [], "今天去{c}花了{amt}元，用{a}結帳。"),
    ]
    l1_income_templates = [
        "{c}入帳",
        "{c}",
    ]
    l1_income_texts = [
        "收到{c}{amt}元匯到{a}了。",
        "{c}{amt}元已經入{a}帳戶。",
        "{c}{amt}元入帳到{a}。",
        "今天{a}收到了{c}{amt}元。",
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
            desc = desc_tmpl.format(c=cat, a=acc, amt=amt, place=place, store=store, item=item)
        else:
            amt = random.choice([1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 22000, 25000, 28000, 30000, 32000, 35000, 38000, 40000, 42000, 45000, 48000, 50000, 52000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 100000])
            desc_tmpl = random.choice(l1_income_templates)
            text_tmpl = random.choice(l1_income_texts)
            user_text = text_tmpl.format(c=cat, a=acc, amt=amt)
            desc = desc_tmpl.format(c=cat, a=acc, amt=amt)
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
