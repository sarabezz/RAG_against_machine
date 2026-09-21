import json

with open('data/output/search_results/dataset_docs_public.json') as f:
    d = json.load(f)

d['search_results'] = d['search_results'][:3]

with open('data/output/search_results/mini_test.json', 'w') as f:
    json.dump(d, f, indent=2)

print("mini_test.json creado")