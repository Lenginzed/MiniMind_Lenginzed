# Stage 8 Network Diagnose

- Python executable: `D:\anaconda3\envs\YSJAirCombat\python.exe`
- datasets version: `3.1.0`
- huggingface_hub version: `0.36.2`
- requests version: `2.32.3`
- OpenSSL: `OpenSSL 3.0.16 11 Feb 2025`
- HF datasets cache: `C:\Users\user\.cache\huggingface\datasets`
- Cache exists: `True`
- Partial/cache lock sample count: `0`

## Environment Variables

- `HTTP_PROXY`: `None`
- `HTTPS_PROXY`: `None`
- `ALL_PROXY`: `None`
- `HF_ENDPOINT`: `None`
- `HF_HOME`: `None`
- `HF_DATASETS_CACHE`: `None`
- `CURL_CA_BUNDLE`: `None`
- `REQUESTS_CA_BUNDLE`: `None`

## URL Checks

- urllib `https://huggingface.co`: `{'ok': True, 'status': 200, 'elapsed_sec': 4.87, 'body_head': '<!doctype html>\n<html class="">\n\t<head>\n\t\t<meta charset="utf-8" />\n\n\t\t<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no" />\n\n\t\t<meta name="description" content="We'}`
- requests `https://huggingface.co`: `{'ok': False, 'elapsed_sec': 0.354, 'error': 'ProxyError(MaxRetryError("HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: / (Caused by ProxyError(\'Unable to connect to proxy\', SSLError(SSLZeroReturnError(6, \'TLS/SSL connection has been closed (EOF) (_ssl.c:1135)\'))))"))'}`
- urllib `https://huggingface.co/api/datasets/Salesforce/wikitext`: `{'ok': True, 'status': 200, 'elapsed_sec': 5.224, 'body_head': '{"_id":"621ffdd236468d709f18200d","id":"Salesforce/wikitext","author":"Salesforce","sha":"b08601e04326c79dfdd32d625aee71d232d685c3","lastModified":"2024-01-04T16:49:18.000Z","private":false,"gated":fa'}`
- requests `https://huggingface.co/api/datasets/Salesforce/wikitext`: `{'ok': False, 'elapsed_sec': 0.168, 'error': 'ProxyError(MaxRetryError("HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /api/datasets/Salesforce/wikitext (Caused by ProxyError(\'Unable to connect to proxy\', SSLError(SSLZeroReturnError(6, \'TLS/SSL connection has been closed (EOF) (_ssl.c:1135)\'))))"))'}`
- urllib `https://huggingface.co/api/datasets/tatsu-lab/alpaca`: `{'ok': True, 'status': 200, 'elapsed_sec': 0.795, 'body_head': '{"_id":"640f5b2fb63b6f18522d6d44","id":"tatsu-lab/alpaca","author":"tatsu-lab","sha":"dce01c9b08f87459cf36a430d809084718273017","lastModified":"2023-05-22T20:33:36.000Z","private":false,"gated":false,'}`
- requests `https://huggingface.co/api/datasets/tatsu-lab/alpaca`: `{'ok': False, 'elapsed_sec': 0.14, 'error': 'ProxyError(MaxRetryError("HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /api/datasets/tatsu-lab/alpaca (Caused by ProxyError(\'Unable to connect to proxy\', SSLError(SSLZeroReturnError(6, \'TLS/SSL connection has been closed (EOF) (_ssl.c:1135)\'))))"))'}`

## Hugging Face Hub API

- `Salesforce/wikitext`: `{'ok': False, 'elapsed_sec': 0.181, 'error': 'ProxyError(MaxRetryError("HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /api/datasets/Salesforce/wikitext (Caused by ProxyError(\'Unable to connect to proxy\', SSLError(SSLZeroReturnError(6, \'TLS/SSL connection has been closed (EOF) (_ssl.c:1135)\'))))"), \'(Request ID: ffa4702d-263b-48c0-92e5-c4b857a63fff)\')'}`
- `tatsu-lab/alpaca`: `{'ok': False, 'elapsed_sec': 0.117, 'error': 'ProxyError(MaxRetryError("HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /api/datasets/tatsu-lab/alpaca (Caused by ProxyError(\'Unable to connect to proxy\', SSLError(SSLZeroReturnError(6, \'TLS/SSL connection has been closed (EOF) (_ssl.c:1135)\'))))"), \'(Request ID: dda6e993-c667-45a5-b1b1-c61acc5bb5b9)\')'}`
- `roneneldan/TinyStories`: `{'ok': False, 'elapsed_sec': 0.138, 'error': 'ProxyError(MaxRetryError("HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /api/datasets/roneneldan/TinyStories (Caused by ProxyError(\'Unable to connect to proxy\', SSLError(SSLZeroReturnError(6, \'TLS/SSL connection has been closed (EOF) (_ssl.c:1135)\'))))"), \'(Request ID: 800471b8-876d-41fa-8679-910474248698)\')'}`

## SSL

```json
{
  "default_verify_paths": {
    "cafile": "D:\\anaconda3\\Library\\ssl\\cacert.pem",
    "capath": null,
    "openssl_cafile_env": "SSL_CERT_FILE",
    "openssl_cafile": "C:\\Program Files\\Common Files\\ssl/cert.pem",
    "openssl_capath_env": "SSL_CERT_DIR",
    "openssl_capath": "C:\\Program Files\\Common Files\\ssl/certs"
  },
  "openssl_version": "OpenSSL 3.0.16 11 Feb 2025"
}
```
