from crypto_module import hybrid_encrypt_full, hybrid_decrypt


def test_cycle_complet():
    message = "Message ultra-secret HPQS-AI !"

    for kem_level in [512, 768, 1024]:
        print("\n==============================")
        print(f"TEST INTÉGRATION ML-KEM-{kem_level}")
        print("==============================")

        bundle, rsa_private_key, rsa_public_key, mlkem_secret_key, mlkem_public_key = hybrid_encrypt_full(
            message,
            kem_level
        )

        decrypted_message = hybrid_decrypt(
            bundle,
            rsa_private_key,
            mlkem_secret_key,
            kem_level
        )

        print("Message original  :", message)
        print("Message déchiffré :", decrypted_message)

        if message == decrypted_message:
            print("TEST RÉUSSI : chiffrement → déchiffrement fonctionne")
        else:
            print("TEST ÉCHOUÉ")


if __name__ == "__main__":
    test_cycle_complet()