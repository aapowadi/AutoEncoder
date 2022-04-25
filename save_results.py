def save_csv(saver_log_folder, epoch, train_losses, test_losses=0, train_accuracy=0, test_accuracy=0,
             cont=True):

    if (epoch == 0 and not cont):
        file = open(saver_log_folder + "/accuracy_results.csv", "w")
        file.close()
        file = open(saver_log_folder + "/losses.csv", "w")
        file.close()

    file = open(saver_log_folder + "/losses.csv", "a")
    file_str = str(float(epoch)) + "," + str(float(train_losses)) + "," \
               + str(float(test_losses)) + "\n"
    file.write(file_str)
    file.close()

    file = open(saver_log_folder + "/accuracy_results.csv", "a")

    out = str(float(epoch)) + "," + str(train_accuracy) + "," + str(
        test_accuracy) + "\n"
    file.write(out)
    file.close()
