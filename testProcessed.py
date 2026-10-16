# import os
# import pathlib
#
# BASE = pathlib.Path(r"C:\Users\niloz\PycharmProjects\Tourism_ProtoNet")
# for d in ["processed_japan_v2", "processed_asia_v1"]:
#     print(f"\n{d}:")
#     for f in os.listdir(BASE / d):
#         print(f"  {f}")



import pandas as pd, pickle
# pois = pd.read_parquet(r"C:\Users\niloz\PycharmProjects\Tourism_ProtoNet\processed_asia_v1\pois.parquet")
pois = pd.read_parquet(r"/Users/aran/Downloads/Proposal/Model/Tourism_ProtoNet/processed_asia_v1\pois.parquet")
print(pois.columns.tolist())
print(pois.head(2))
print(pois.shape)

# pois_japan = pd.read_parquet(r"C:\Users\niloz\PycharmProjects\Tourism_ProtoNet\processed_japan_v2\pois.parquet")
pois_japan = pd.read_parquet(r"/Users/aran/Downloads/Proposal/Model/Tourism_ProtoNet/processed_japan_v2\pois.parquet")

print(pois_japan.columns.tolist())


# grep -r "category_id" C:\Users\niloz\PycharmProjects\Tourism_ProtoNet\src\