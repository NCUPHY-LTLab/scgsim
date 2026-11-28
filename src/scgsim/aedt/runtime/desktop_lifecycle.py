"""Release the transaction-owned Desktop after result capture.

Stop-save settlement determines whether to close work or detach for recovery.
"""

from __future__ import annotations

from typing import Any, cast


def release_owned_desktop(
    desktop: Any, receipt: dict[str, Any], result: dict[str, Any] | None
) -> None:
    """Close saved work; detach without closing when stop-save remains unresolved."""
    preserve_desktop = False
    try:
        preserve_desktop = (
            result is not None
            and "termination" in result
            and (
                result["termination"]["control"]
                .get("native_settlement", {})
                .get("status")
                != "inactive"
                or result["termination"]["control"]
                .get("request_readback", {})
                .get("status")
                == "unavailable"
                or result["save"]["ok"] is not True
            )
        )
        if preserve_desktop:
            result = cast(dict[str, Any], result)
            receipt["release"] = {
                "ok": False,
                "reason": "save_failed"
                if result["save"].get("attempted") and not result["save"]["ok"]
                else "native_not_confirmed_inactive",
                "preserved_desktop": result["termination"]["control"]["desktop"],
                "ownership": "unresolved",
            }
            detached = desktop.release_desktop(
                close_projects=False, close_on_exit=False
            )
            receipt["release"]["detach"] = {
                "ok": detached is True,
                "native_return": detached,
            }
        else:
            released = bool(
                desktop.release_desktop(close_projects=True, close_on_exit=True)
            )
            receipt["release"] = {"ok": released}
            if not released:
                raise RuntimeError("owned AEDT Desktop release returned false")
    except Exception as exc:  # noqa: BLE001 -- receipt records release failure.
        receipt["release"] = {
            **receipt.get("release", {}),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if preserve_desktop:
            receipt["release"]["detach"] = {
                "ok": False,
                "error": receipt["release"]["error"],
            }
        raise
