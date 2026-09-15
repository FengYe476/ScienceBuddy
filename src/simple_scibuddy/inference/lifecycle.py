"""Keep each HTTP session inside the event loop that created it."""

import asyncio


class InferenceLifecycle:
    def _get_new_inference_client(self):
        from skyrl.backends.skyrl_train.inference_servers.setup import build_new_inference_client

        colocated = self.cfg.trainer.placement.colocate_all
        client, setup = build_new_inference_client(
            self.cfg, self.tokenizer, placement_group=self.colocate_pg if colocated else None
        )
        self._inference_router = setup.router
        self._server_groups = setup.server_groups
        self._prefill_server_groups = setup.prefill_server_groups
        self._decode_server_groups = setup.decode_server_groups

        async def sleep_and_close():
            try:
                await client.sleep(level=1 if getattr(self, "evaluation_only", False) else 2)
            finally:
                # Upstream uses a temporary asyncio.run here, then starts training
                # on a different loop. Close before that temporary loop goes away.
                await client.aclose()

        if colocated:
            asyncio.run(sleep_and_close())
        return client
