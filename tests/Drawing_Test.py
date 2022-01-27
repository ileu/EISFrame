from matplotlib import pyplot as plt

from eisplottingtool.drawing import draw_circuit


def main():
    # circuit1 = 'p(R,R,R,R)-p(R,R,R)'
    # circuit2 = 'R-p(C,Ws-p(R,R))-p(R,R,R)'
    # circuit3 = 'R-p(R-R,Ws-p(R,R-p(R,R)),R)-R-p(C,C,C,C)-R'
    # circuit4 = 'R-p(R-p(p(p(R,R),C-R-CPE),R),R,R,R,R,R,R,R,' \
    #            'CPE)-R-R-R-R-R-R-R-p(R,R,C)'
    # circuit5 = 'p(p(R,R),R)'
    # circuits = [circuit1, circuit2, circuit3, circuit4, circuit5]
    circuit1 = "R-p(R1,CPE)-Ws"
    circuit2 = "R-Ws"
    circuits = [circuit1, circuit2]

    path1 = r"G:\Collaborators\Sauter Ulrich\Projects\EIS Tail\Data"
    path2 = r"C:\Users\ueli\Desktop\Sauter Ulrich\Projects\EIS Tail\Data"

    path = path1

    for circ in circuits:
        print(25 * '-')
        print(f"{circ=}")
        drawing = draw_circuit(circ, color_dict={"R1": "blue", "Ws": "green"}, lw=10)
        drawing.draw(show=False)
        backend = plt.get_backend()
        if backend == "QtAgg":
            fm = plt.get_current_fig_manager()
            fm.window.showMaximized()
        elif backend == "TkAgg":
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
        plt.show()


if __name__ == "__main__":
    main()
