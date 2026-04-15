class LargeProcessor:
    """Class with a method exceeding 150 lines."""

    def process_all(self, data):
        """Process all data through multiple stages."""
        # Stage 1: Validation
        if not data:
            return None
        validated = []
        for item in data:
            if item.get("type") == "A":
                validated.append(item)
            elif item.get("type") == "B":
                validated.append(item)
            elif item.get("type") == "C":
                validated.append(item)
            else:
                validated.append({"type": "unknown", **item})

        # Stage 2: Normalization
        normalized = []
        for item in validated:
            name = item.get("name", "")
            value = item.get("value", 0)
            normalized.append({
                "name": name.strip().lower(),
                "value": float(value),
                "type": item["type"],
            })

        # Stage 3: Grouping
        groups = {}
        for item in normalized:
            key = item["type"]
            if key not in groups:
                groups[key] = []
            groups[key].append(item)

        # Stage 4: Aggregation
        aggregated = {}
        for key, items in groups.items():
            total = sum(i["value"] for i in items)
            count = len(items)
            avg = total / count if count else 0
            aggregated[key] = {
                "total": total,
                "count": count,
                "average": avg,
            }

        # Stage 5: Enrichment
        enriched = {}
        for key, agg in aggregated.items():
            enriched[key] = {
                **agg,
                "percentage": agg["total"] / max(sum(a["total"] for a in aggregated.values()), 1) * 100,
                "rank": 0,
            }

        # Stage 6: Ranking
        sorted_keys = sorted(enriched.keys(), key=lambda k: enriched[k]["total"], reverse=True)
        for rank, key in enumerate(sorted_keys, 1):
            enriched[key]["rank"] = rank

        # Stage 7: Formatting
        formatted = []
        for key in sorted_keys:
            info = enriched[key]
            formatted.append({
                "category": key,
                "total_value": round(info["total"], 2),
                "count": info["count"],
                "average": round(info["average"], 2),
                "percentage": round(info["percentage"], 1),
                "rank": info["rank"],
            })

        # Stage 8: Validation of output
        for item in formatted:
            assert item["count"] > 0
            assert item["total_value"] >= 0
            assert 0 <= item["percentage"] <= 100
            assert item["rank"] > 0

        # Stage 9: Summary
        summary = {
            "total_categories": len(formatted),
            "total_items": sum(f["count"] for f in formatted),
            "grand_total": sum(f["total_value"] for f in formatted),
        }

        # Stage 10: Logging
        for item in formatted:
            print(f"Category {item['category']}: {item['total_value']}")

        # Stage 11: Additional processing
        for item in formatted:
            item["label"] = f"{item['category']} ({item['count']} items)"
            item["description"] = f"Total: {item['total_value']}, Avg: {item['average']}"

        # Stage 12: Final checks
        if not formatted:
            return {"summary": summary, "data": []}

        # Stage 13: Sort by rank
        formatted.sort(key=lambda x: x["rank"])

        # Stage 14: Add metadata
        for i, item in enumerate(formatted):
            item["index"] = i
            item["is_top"] = i < 3

        # Stage 15: Build response
        response = {
            "summary": summary,
            "data": formatted,
            "metadata": {
                "processed": True,
                "version": "1.0",
            },
        }

        # Stage 16: Cleanup
        for item in response["data"]:
            if "label" in item:
                item["display_name"] = item.pop("label")

        # Stage 17: Final validation
        assert response["summary"]["total_categories"] == len(response["data"])
        assert all("display_name" in d for d in response["data"])

        # Stage 18: Cache result
        self._last_result = response

        # Stage 19: Notify observers
        self._notify("process_complete", response["summary"])

        # Stage 20: Export statistics
        stats = {}
        for item in response["data"]:
            stats[item["category"]] = {
                "total": item["total_value"],
                "avg": item["average"],
                "pct": item["percentage"],
            }

        # Stage 21: Generate report
        report_lines = []
        for cat, info in stats.items():
            report_lines.append(f"{cat}: total={info['total']}, avg={info['avg']}")

        # Stage 22: Write output
        response["report"] = "\n".join(report_lines)
        response["stats"] = stats

        # Stage 23: Archive
        archived = {
            "version": 2,
            "data": response["data"][:],
            "timestamp": None,
        }
        self._archive = archived

        # Stage 24: Final cleanup
        for key in list(response.keys()):
            if key.startswith("_"):
                del response[key]

        # Stage 25: Return
        return response

    def _notify(self, event, data):
        """Notify observers of an event."""
        pass
