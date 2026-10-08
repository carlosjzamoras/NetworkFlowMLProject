from base_values import BaseValues


def main():
    base_values = BaseValues()

    # Properties are accessed WITHOUT parentheses — city_list, edges,
    # city_dist, cap_dict, demand_dict, supply_dict, price_dict are all
    # attributes, not methods, once computed.
    print("City list (first 5):", base_values.city_list[:5])
    print("Number of cities:", len(base_values.city_list))

    print("\nEdges (first 5):", base_values.edges[:5])
    print("Number of edges (bidirectional):", len(base_values.edges))

    print("\nCity distances (first 3 pairs):")
    for i, (pair, dist) in enumerate(base_values.city_dist.items()):
        if i >= 3:
            break
        print(f"  {pair}: {dist:.1f} miles")

    print("\nCapacity dict (first 3 entries):")
    for i, (pair, cap) in enumerate(base_values.cap_dict.items()):
        if i >= 3:
            break
        print(f"  {pair}: {cap}")

    print("\nDemand dict (first 3 entries):")
    for i, (key, val) in enumerate(base_values.demand_dict.items()):
        if i >= 3:
            break
        print(f"  {key}: {val}")

    print("\nSupply dict (first 3 entries):")
    for i, (key, val) in enumerate(base_values.supply_dict.items()):
        if i >= 3:
            break
        print(f"  {key}: {val}")

    print("\nPrice dict (first 3 entries):")
    for i, (key, val) in enumerate(base_values.price_dict.items()):
        if i >= 3:
            break
        print(f"  {key}: {val}")

    # Confirm caching is actually working: access twice, time the second
    # call — it should be near-instant since it's just reading the cached
    # value off the instance, not recomputing.
    import time
    start = time.perf_counter()
    _ = base_values.supply_dict  # first access already happened above,
                                   # so this is testing the cache hit
    elapsed = time.perf_counter() - start
    print(f"\nSecond access to supply_dict took {elapsed:.6f} seconds (should be ~0)")


if __name__ == "__main__":
    main()