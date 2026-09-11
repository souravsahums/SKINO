"""Resolve the Azure ML workspace this code should target.

On a compute instance we want to reuse *that instance's own* workspace,
subscription and resource group with nothing hand-set. Resolution order (first
complete match wins):

  1. explicit ``AML_*`` env vars (manual override, kept for back-compat)
  2. the compute instance's own context (``AZUREML_ARM_*`` env vars, which Azure
     ML sets automatically on every compute instance and inside every job)
  3. the ``config.json`` shipped on the compute instance (``MLClient.from_config``)
"""
from __future__ import annotations

import os


def resolve_workspace():
    """Return ``(subscription, resource_group, workspace)`` from env, any may be None."""
    sub = os.environ.get("AML_SUBSCRIPTION_ID") or os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    rg = os.environ.get("AML_RESOURCE_GROUP") or os.environ.get("AZUREML_ARM_RESOURCEGROUP")
    ws = os.environ.get("AML_WORKSPACE") or os.environ.get("AZUREML_ARM_WORKSPACE_NAME")
    return sub, rg, ws


def workspace_label():
    """Human-readable target for logging before a client exists (e.g. in --dry-run)."""
    sub, rg, ws = resolve_workspace()
    if ws and rg:
        return f"{ws}  (rg={rg})"
    return "(auto-detected from the compute instance's config.json)"


def get_ml_client(credential):
    """An ``MLClient`` bound to the compute instance's own workspace.

    Uses the ``AML_*`` / ``AZUREML_ARM_*`` env vars when present, else falls back
    to ``MLClient.from_config`` which reads the config the compute instance ships.
    """
    from azure.ai.ml import MLClient

    sub, rg, ws = resolve_workspace()
    if sub and rg and ws:
        return MLClient(credential, sub, rg, ws)
    return MLClient.from_config(credential)
