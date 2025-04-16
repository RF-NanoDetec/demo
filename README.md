
# Frequency‑Domain Filtering Demo

Interactive Streamlit app that shows how band‑/low‑/high‑pass filtering in the frequency domain
cleans up 1 ms Gaussian pulses buried in noise.

## Local run

```bash
pip install -r requirements.txt
streamlit run filter_demo.py
```

## Deploy on Streamlit Community Cloud

1. Fork or clone this repo, or just upload it to a new public GitHub repository.
2. Go to **https://share.streamlit.io** and click **“New app”**.
3. Select your repository and set **Main file** to `filter_demo.py`.
4. Click **Deploy** — in ~1 minute your app will be live at  
   `https://<your‑handle>-filter-demo.streamlit.app`.

## Screenshot

![Screenshot](https://raw.githubusercontent.com/streamlit/streamlit/master/examples/demos/media/demo-gif.gif)
