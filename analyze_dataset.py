import json
import os
import sys
import re
from collections import Counter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

ACCOUNT_KEYWORDS = [
    '現金', '付現', '信用卡', '刷卡', '卡', '悠遊卡', '一卡通', 'icash', '悠遊付',
    '街口', 'LINE Pay', 'Apple Pay', 'Google Pay', 'Samsung Pay', '台灣Pay', '全支付',
    '全盈', 'OPEN錢包', 'Pi錢包', '橘子支付', '歐付寶', '郵局', '銀行', '存款', '帳戶',
    '外幣', '加密貨幣', 'Richart', 'LINE Bank', '將來銀行', '樂天', 'O-Bank',
    '轉帳', '匯款', '扣款', '提款', '交割戶', '複委託', '美元', '日圓', '台新', '國泰',
    '富邦', '中信', '玉山', '永豐', '聯邦', '元大', '星展', '滙豐', '兆豐', '第一', '華南',
    '彰化', '合庫', '渣打', '大戶', 'iLeo', 'KOKO', 'SnY'
]

def analyze_jsonl(filepath):
    if not os.path.exists(filepath):
        return None
    total = 0
    difficulties = Counter()
    categories = Counter()
    accounts = Counter()
    short_count = 0
    no_account_kw_count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            data = json.loads(line)
            diff = data.get('difficulty_level', 'Unknown')
            difficulties[diff] += 1
            
            msgs = data.get('messages', [])
            user_msg = next((m['content'] for m in msgs if m['role'] == 'user'), '')
            asst_msg = next((m['content'] for m in msgs if m['role'] == 'assistant'), '')
            
            if len(user_msg) <= 12:
                short_count += 1
                
            has_kw = any(kw in user_msg for kw in ACCOUNT_KEYWORDS)
            if not has_kw:
                no_account_kw_count += 1

            m_cat = re.search(r'"category"\s*:\s*"([^"]+)"', asst_msg)
            if m_cat:
                categories[m_cat.group(1)] += 1
            else:
                m_cat2 = re.search(r'\[CAT\]\s*([^\[<]+)', asst_msg)
                if m_cat2:
                    categories[m_cat2.group(1).strip()] += 1

            m_acc = re.search(r'"account"\s*:\s*"([^"]+)"', asst_msg)
            if m_acc:
                accounts[m_acc.group(1)] += 1
            else:
                m_acc2 = re.search(r'\[ACC\]\s*([^\[<]+)', asst_msg)
                if m_acc2:
                    accounts[m_acc2.group(1).strip()] += 1

    return {
        'total': total,
        'difficulties': dict(difficulties),
        'categories': dict(categories.most_common(10)),
        'accounts': dict(accounts.most_common(10)),
        'short_count': short_count,
        'no_account_kw_count': no_account_kw_count
    }

def main():
    print('=== 1. 當前生成資料集狀態 (train_strata.jsonl) ===')
    info_train = analyze_jsonl('dataset/train_strata.jsonl')
    if info_train:
        print(f'總訓練樣本數: {info_train["total"]}')
        print(f'難度分佈: {info_train["difficulties"]}')
        print(f'短描述 (<= 12字) 數量: {info_train["short_count"]} ({info_train["short_count"]/info_train["total"]*100:.2f}%)')
        print(f'未提及帳戶關鍵字樣本數: {info_train["no_account_kw_count"]} ({info_train["no_account_kw_count"]/info_train["total"]*100:.2f}%)')
        print(f'Top 10 分類: {info_train["categories"]}')
        print(f'Top 10 帳戶: {info_train["accounts"]}')

    print('\n=== 2. 測試集狀態 (test_strata.jsonl) ===')
    info_test = analyze_jsonl('dataset/test_strata.jsonl')
    if info_test:
        print(f'總測試樣本數: {info_test["total"]}')
        print(f'短描述 (<= 12字) 數量: {info_test["short_count"]} ({info_test["short_count"]/info_test["total"]*100:.2f}%)')

    print('\n=== 3. 稽核進度與退件原因分析 ===')
    if os.path.exists('dataset/train_audited.jsonl') and os.path.exists('dataset/rejected_audited.jsonl'):
        pass_cnt = sum(1 for line in open('dataset/train_audited.jsonl', encoding='utf-8') if line.strip())
        rej_cnt = sum(1 for line in open('dataset/rejected_audited.jsonl', encoding='utf-8') if line.strip())
        total_audited = pass_cnt + rej_cnt
        print(f'已稽核總數: {total_audited} / 4000 ({total_audited/4000*100:.1f}%)')
        print(f'  - ✅ 稽核通過數: {pass_cnt} ({pass_cnt/total_audited*100:.2f}%)')
        print(f'  - ❌ 稽核退件數: {rej_cnt} ({rej_cnt/total_audited*100:.2f}%)')
        
        reasons = Counter()
        with open('dataset/rejected_audited.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                r = data.get('reason', '未知原因')
                if '帳戶' in r or '現金' in r:
                    r_type = '帳戶預設對齊失敗'
                elif '日期' in r:
                    r_type = '日期對齊失敗'
                elif '分類' in r or 'type' in r:
                    r_type = '分類/收支類型混淆'
                elif '金額' in r:
                    r_type = '金額數值不一致'
                else:
                    r_type = r[:50]
                reasons[r_type] += 1
        print('退件原因分佈:')
        for r_k, r_v in reasons.most_common(10):
            print(f'  - {r_k}: {r_v} 筆 ({r_v/rej_cnt*100:.1f}%)')

if __name__ == '__main__':
    main()
