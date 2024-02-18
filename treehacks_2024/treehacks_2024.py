"""Welcome to Reflex! This file outlines the steps to create a basic app."""

from rxconfig import config

import reflex as rx

useless = "https://theuselessweb.com/"

class State(rx.State):
    """The app state."""


def index() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.heading("Munch!", size="9"),
            rx.text("No need to think, start munching"),
            rx.button(
                "Start Munching!",
                on_click=lambda: rx.redirect(useless),
                size="4",
            ),
            align="center",
            spacing="7",
            font_size="2em",
        ),
        height="100vh",
    )


app = rx.App()
app.add_page(index)
