from time_parser import TimeParser

def test_now():
    parser = TimeParser()
    queries = [
        "D.Trump现在多少岁",
        "现在几点了",
        "目前的局势",
        "最近三天的新闻"
    ]
    
    print("Testing 'now' queries:")
    for q in queries:
        res = parser.parse(q)
        print(f"Query: {q}")
        print(f"  Has constraint: {res.days is not None}")
        if res.days:
            print(f"  Days: {res.days}")
            print(f"  Expression: {res.time_expression}")
        print("-" * 20)

if __name__ == "__main__":
    test_now()
