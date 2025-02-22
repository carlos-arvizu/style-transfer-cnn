from style_transfer import style_transfer

def main():
    #define paths to images
    content_image = "examples/content/bellas_artes.jpg"
    style_image = "examples/style/starry_night.jpg"
    target_image = "examples/content/bellas_artes.jpg"  # Can also use a white noise image

    #define hyperparameters
    learning_rate = 0.01
    steps = 3000
    alpha = .1 #importance to content image
    beta = 100000000 #importance to style image
    output_name = 'output_2'

    # Run style transfer
    style_transfer(
        content=content_image,
        style=style_image,
        target=target_image,
        lr=learning_rate,
        steps=steps,
        alpha=alpha,
        beta=beta,
        name = output_name
    )

if __name__ == "__main__":
    main()
