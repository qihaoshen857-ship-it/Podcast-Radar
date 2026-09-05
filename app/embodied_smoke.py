"""Real Tk layout checks, also run inside the installed Windows executable."""
from __future__ import annotations


def exercise_embodied_ui(app, api) -> dict:
    root = app.root
    topic = api.EMBODIED_TOPIC
    root.deiconify()
    app.current_topic = topic
    app.download_title_var.set(api.topic_download_title(topic))
    app.update_topic_filter_buttons()
    sample = api.VideoItem(
        video_id="robot-ui-smoke",
        title="Humanoid Robot Foundation Models: A Complete Conversation on Vision Language Action Policies, Dexterous Hands, Manufacturing Costs and Real Factory Deployment",
        description_text="Founder interview: VLA policies and dexterous hands; 200 robots delivered.",
        duration_text="45:00", webpage_url="https://example.com/ui-test",
        source_name="Layout test fixture", topic=topic,
    )
    app.video_items = [sample]
    app.video_map = {sample.video_id: sample}
    app.render_cards()
    sizes = []
    for width, height in ((1080, 720), (1280, 820)):
        root.geometry(f"{width}x{height}")
        root.update_idletasks()
        app.layout_topic_header()
        root.update_idletasks()
        ordered = [app.topic_filter_buttons[key] for key in api.TOPIC_FILTER_LABELS]
        rects = [(w.winfo_rootx(), w.winfo_rooty(), w.winfo_width(), w.winfo_height()) for w in ordered]
        for idx, (x, y, w, h) in enumerate(rects):
            assert w > 30 and h > 20, (idx, rects)
            assert root.winfo_rootx() <= x and x + w <= root.winfo_rootx() + root.winfo_width(), rects
            if idx:
                assert rects[idx-1][0] + rects[idx-1][2] <= x, rects
        heading = app.topic_heading
        filters = app.topic_filters
        assert (heading.winfo_rootx() + heading.winfo_width() <= filters.winfo_rootx()
                or heading.winfo_rooty() + heading.winfo_height() <= filters.winfo_rooty()), "header overlap"
        assert app.download_title_var.get() == "今日雷达 · 具身智能"
        button = app.topic_filter_buttons[topic]
        assert button.itemcget(button.find_all()[0], "fill") == api.COLOR_TEXT
        sizes.append({"width": width, "height": height, "buttons_visible": True, "no_overlap": True,
                      "filter_row": int(filters.grid_info()["row"])})
    app.begin_feed_refresh(topic, showing_cache=False)
    root.update_idletasks()
    assert "正在刷新 · 具身智能" in app.refresh_indicator_var.get()
    assert app.progress_percent_var.get() == "刷新中"
    app.finish_feed_refresh(success=True)
    root.update_idletasks()
    return {"ok": True, "viewports": sizes, "selected_black": True, "refresh_feedback": True}
