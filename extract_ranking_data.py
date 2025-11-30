import json
import re
from main import build_search_client, build_llm_client
from smart_orchestrator import SmartSearchOrchestrator

# 加载配置
with open("config.json", "r") as f:
    config = json.load(f)

# 构建客户端
search_client = build_search_client(config)
llm_client = build_llm_client(config)

# 创建orchestrator
orchestrator = SmartSearchOrchestrator(llm_client=llm_client, search_client=search_client)

# 执行搜索
query = '香港中文大學與香港科技大學最近10年的QS排名對比'
result = orchestrator.answer(query, num_search_results=20)

# 提取排名数据
print('=== 提取排名数据 ===')
cuhk_rankings = {}
hkust_rankings = {}

for hit in result['search_hits']:
    title = hit.get('title', '')
    snippet = hit.get('snippet', '')
    text = f"{title} {snippet}"
    
    # 提取年份和排名信息
    year_rank_pattern = r'(20\d{2})[^0-9]*#?(\d{1,3})'
    matches = re.findall(year_rank_pattern, text)
    
    for year, rank in matches:
        year = int(year)
        rank = int(rank)
        
        # 检查是否与CUHK相关
        if ('cuhk' in text.lower() or 'chinese university of hong kong' in text.lower() or 
            '香港中文' in text):
            if year not in cuhk_rankings or rank < cuhk_rankings[year]:
                cuhk_rankings[year] = rank
                print(f"CUHK {year}: #{rank}")
        
        # 检查是否与HKUST相关
        if ('hkust' in text.lower() or 'hong kong university of science and technology' in text.lower() or 
            '香港科技' in text):
            if year not in hkust_rankings or rank < hkust_rankings[year]:
                hkust_rankings[year] = rank
                print(f"HKUST {year}: #{rank}")

# 打印提取的排名数据
print('\n=== CUHK排名数据 ===')
for year in sorted(cuhk_rankings.keys()):
    print(f"{year}: #{cuhk_rankings[year]}")

print('\n=== HKUST排名数据 ===')
for year in sorted(hkust_rankings.keys()):
    print(f"{year}: #{hkust_rankings[year]}")

# 尝试更宽松的匹配模式
print('\n=== 尝试更宽松的匹配模式 ===')

# 查找所有包含排名信息的文本
for i, hit in enumerate(result['search_hits'], 1):
    title = hit.get('title', '')
    snippet = hit.get('snippet', '')
    text = f"{title} {snippet}"
    
    # 查找包含排名的行
    if ('rank' in text.lower() or '排名' in text) and ('20' in text):
        print(f"\n结果 {i}:")
        print(f"标题: {title}")
        print(f"摘要: {snippet[:300]}...")
        
        # 尝试提取排名信息
        lines = text.split('.')
        for line in lines:
            if ('rank' in line.lower() or '排名' in line) and ('20' in line):
                print(f"  排名行: {line.strip()}")