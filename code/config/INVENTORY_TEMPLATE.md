# Operator-supplied private inventory

Live acquisition reads a JSON object with a `ports` array. Replace every
placeholder and independently verify each SSH host key before acquisition.
The example address belongs to the documentation range and is not a lab
endpoint. Analysis and simulation commands do not use this file.

```json
{
  "ports": [
    {
      "port": 1,
      "ip": "192.0.2.11",
      "username": "REPLACE_WITH_LOCAL_ACCOUNT",
      "host_key_sha256": "SHA256:REPLACE_WITH_VERIFIED_PUBLIC_HOST_KEY_FINGERPRINT",
      "ssh_login": true
    }
  ]
}
```

The registered new cohort uses port identifiers
`1, 3, 4, 5, 7, 8, 9, 10, 12, 17, 19, 21, 22, 23`.
Provide one verified row for each endpoint. The cohort and rate topology are
part of the experimental design, rather than inferred from IP addresses.
The source verifies pinned keys and does not automatically accept a changed
host identity.

Set `EDGE_SSH_KEY` to the operator's private-key path and set
`EDGE_SWITCH_HOST` and `EDGE_SWITCH_USERNAME` for the switch. The switch
password is entered at a hidden prompt. Keep the inventory and private key
outside the public artifact and outside any Git repository.
