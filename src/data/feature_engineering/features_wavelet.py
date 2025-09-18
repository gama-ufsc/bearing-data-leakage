from pywt import wavedec


def wavelet_features_paderborn(df, signal_column="denoised_signal", additional_name=""):
    coeffs = df[signal_column].apply(lambda x: wavedec(x, wavelet="bior3.7", level=3))

    df = df.assign(
        **{
            f"{additional_name}cA3": coeffs.apply(lambda x: sum(x[0])),
            f"{additional_name}cD3": coeffs.apply(lambda x: sum(x[1])),
            f"{additional_name}cD2": coeffs.apply(lambda x: sum(x[2])),
            f"{additional_name}cD1": coeffs.apply(lambda x: sum(x[3])),
        }
    )

    return df
