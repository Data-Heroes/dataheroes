
class CoresetTreeService(CoresetServiceBase):
    """
    Service class for creating and working with a coreset tree

    Parameters:
        data_manager: DataManagerBase subclass, optional

        data_params: DataParams, optional
            Preprocessing information.

        coreset_size: int/dict, required
            Represent the coreset size of each node in the tree.
            dict {class: size}: control the number of samples to choose for each class.

        sample_all: iterable, optional
            Classification only, a list of classes for them all samples should be taken.
            When provided, coreset_size must be an int.

        coreset_params: CoresetParams or dict, optional
            Corset algorithm specific parameters.

        node_train_function: Callable, optional
            method for training model at tree node level

        node_train_function_params: dict, optional
            kwargs to be used when calling node_train_function

        node_metadata_func: callable, optional
            A method for storing user meta data on each node

        working_directory: str, path, optional
            Local directory where intermediate data is stored.

        cache_dir: str, path, optional
            For internal use when loading a saved service.
    """
    pass

    def __init__(
            self,
            *,
            data_manager: DataManagerT = None,
            data_params: Union[DataParams, dict] = None,
            coreset_size: Union[int, dict],
            coreset_params: Union[CoresetParams, dict] = None,
            sample_all: Iterable = None,
            class_size: Dict[Any, int] = None,
            working_directory: Union[str, os.PathLike] = None,
            cache_dir: Union[str, os.PathLike] = None,
            node_train_function: Callable[[np.ndarray, np.ndarray, np.ndarray], Any] = None,
            node_train_function_params: dict = None,
            node_metadata_func: Callable[
                [Tuple[np.ndarray], np.ndarray, Union[list, None]], Union[list, dict, None]
            ] = None,
    ):
        pass

    @telemetry
    def build_from_file(
            self,
            file_path: Union[Union[str, os.PathLike], Iterable[Union[str, os.PathLike]]],
            target_file_path: Union[Union[str, os.PathLike], Iterable[Union[str, os.PathLike]]] = None,
            *,
            reader_f=pd.read_csv,
            reader_kwargs: dict = None,
            sample_size: int = None,
            coreset_by: Union[Callable, str, list] = None
    ) -> 'CoresetTreeService':
        """
        Create a coreset tree based on the data taken from a local storage.
        Parameters:
            file_path: file, list of files, directory, list of directories.
                Path(s) to the place where data is stored.
                Data includes features, may include labels and may include indices.

            target_file_path: file, list of files, directory, list of directories, optional
                Use when files are split to features and labels.
                Each file should include only one column.

            reader_f: pandas like read method, optional, default pandas read_csv
                For example, to read excel files use pandas read_excel.

            reader_kwargs: dict, optional
                Keyword arguments used when calling reader_f method.

            sample_size: int
                The number of instances used when creating a coreset node in the tree.
                sample_size=0: nodes are created based on input chunks

            coreset_by: function, label, or list of labels, optional
                Split the data by the key.
                When provided, sample_size input is ignored.

        Returns:
            self
        """
        pass

    @telemetry
    def partial_build_from_file(
            self,
            file_path: Union[Union[str, os.PathLike], Iterable[Union[str, os.PathLike]]],
            target_file_path: Union[Union[str, os.PathLike], Iterable[Union[str, os.PathLike]]] = None,
            *,
            reader_f=pd.read_csv,
            reader_kwargs: dict = None,
            sample_size: int = None,
            coreset_by: Union[Callable, str, list] = None
    ) -> 'CoresetTreeService':
        """
        Adding new samples to a coreset tree based on the data taken from local storage.
        Parameters:
            file_path: file, list of files, directory, list of directories.
                Path(s) to the place where data is stored.
                Data includes features, may include labels and may include indices.

            target_file_path: file, list of files, directory, list of directories, optional
                Use when files are split to features and labels.
                Each file should include only one column.

            reader_f: pandas like read method, optional, default pandas read_csv
                For example, to read excel files use pandas read_excel.

            reader_kwargs: dict, optional
                Keyword arguments used when calling reader_f method.

            sample_size: int, optional, default previous used sample_size
                The number of instances used when creating a coreset node in the tree.
                sample_size=0: nodes are created based on input chunks.

            coreset_by: function, label, or list of labels, optional
                Split the data by the key.
                When provided, sample_size input is ignored.

        Returns:
            self
        """
        pass

    @telemetry
    def build_from_df(
            self,
            datasets: Union[Iterator[pd.DataFrame], pd.DataFrame],
            target_datasets: Union[Iterator[pd.DataFrame], pd.DataFrame] = None,
            *,
            sample_size=None,
            coreset_by: Union[Callable, str, list] = None
    ) -> 'CoresetTreeService':
        """
        Create a coreset tree from pandas DataFrame(s).

        Parameters:
            datasets: pandas DataFrame or DataFrame iterator
                Data includes features, may include labels and may include indices.

            target_datasets: pandas DataFrame or DataFrame iterator, optional
                Use when data is split to features and labels.
                Should include only one column.

            sample_size: int
                The number of instances used when creating a coreset node in the tree.
                sample_size=0: nodes are created based on input chunks

            coreset_by: function, label, or list of labels, optional
                Split each dataset by the key.
                When provided, sample_size input is ignored.

        Returns:
            self
        """
        pass

    def partial_build_from_df(
            self,
            datasets: Union[Iterator[pd.DataFrame], pd.DataFrame],
            target_datasets: Union[Iterator[pd.DataFrame], pd.DataFrame] = None,
            *,
            sample_size=None,
            coreset_by: Union[Callable, str, list] = None
    ) -> 'CoresetTreeService':
        """
        Adding new samples to a coreset tree based on the pandas DataFrame iterator.

        Parameters:
            datasets: pandas DataFrame or DataFrame iterator
                Data includes features, may include labels and may include indices.

            target_datasets: pandas DataFrame or DataFrame iterator, optional
                Use when data is split to features and labels.
                Should include only one column.

            sample_size: int, optional, default previous used sample_size
                The number of instances used when creating a coreset node in the tree.
                sample_size=0: nodes are created based on input chunks

            coreset_by: function, label, or list of labels, optional
                Split each dataset by the key.
                When provided, sample_size input is ignored.

        Returns:
            self
        """
        pass

    @telemetry
    def build(
            self,
            X: Union[Iterable, Iterable[Iterable]],
            y: Union[Iterable[Any], Iterable[Iterable[Any]]] = None,
            indices: Union[Iterable[Any], Iterable[Iterable[Any]]] = None,
            *,
            sample_size=None,
            coreset_by: Union[Callable, str, list] = None
    ) -> 'CoresetTreeService':
        """
        Create a coreset tree from parameters X, y and indices.

        Parameters:
            X: array like or iterator of arrays like
                an array or an iterator of features

            y: array like or iterator of arrays like, optional
                an array or an iterator of targets

            indices: array like or iterator of arrays like, optional
                an array or an iterator with indices of X

            sample_size: int
                The number of instances used when creating a coreset node in the tree.
                sample_size=0: nodes are created based on input chunks

            coreset_by: function, label, or list of labels, optional
                Split each dataset by the key.
                When provided, sample_size input is ignored.

        Returns:
            self
        """
        pass

    def partial_build(
            self,
            X: Union[Iterable, Iterable[Iterable]],
            y: Union[Iterable[Any], Iterable[Iterable[Any]]] = None,
            indices: Union[Iterable[Any], Iterable[Iterable[Any]]] = None,
            *,
            sample_size=None,
            coreset_by: Union[Callable, str, list] = None
    ) -> 'CoresetTreeService':
        """
        Adding new samples to a coreset tree from parameters X, y and indices.

        Parameters:
            X: array like or iterator of arrays like
                an array or an iterator of features

            y: array like or iterator of arrays like, optional
                an array or an iterator of targets

            indices: array like or iterator of arrays like, optional
                an array or an iterator with indices of X

            sample_size: int
                The number of instances used when creating a coreset node in the tree.
                sample_size=0: nodes are created based on input chunks

            coreset_by: function, label, or list of labels, optional
                Split each dataset by the key.
                When provided, sample_size input is ignored.

        Returns:
            self
        """
        pass

    @telemetry
    def build_from_tensorflow_dataset(
            self,
            dataset: (Any, Any),
            *,
            sample_size: int = None,
            coreset_by: Union[Callable, str, list] = None
    ) -> 'CoresetTreeService':
        """
        create a coreset tree based on the tf.data.Dataset

        Parameters:
            dataset: tuple (tf.data.Dataset, tfds.core.DatasetInfo)
            sample_size: nodes are created based on input chunks
                default: sample_size is decided internally
            coreset_by: function, label, or list of labels, optional
                split each dataset by key
        """
        pass

    def partial_build_from_tensorflow_dataset(
            self,
            dataset: (Any, Any),
            *,
            sample_size: int = None,
            coreset_by: Union[Callable, str, list] = None
    ):
        """
        Applying new samples to a coreset tree based on the tf.data.Dataset

        Parameters:
            dataset: tuple (tf.data.Dataset, tfds.core.DatasetInfo)
            sample_size: nodes are created based on input chunks
                default: sample_size is decided internally
            coreset_by: function, label, or list of labels, optional
                split each dataset by key
        """
        pass


    @telemetry
    def build_from_tensor(
            self,
            dataset: Any,
            *,
            sample_size: int = None,
            coreset_by: Union[Callable, str, list] = None
    ):
        """
        create a coreset tree based on the torch.Tensor

        Parameters:
            dataset: torch.Tensor
            sample_size: nodes are created based on input chunks
                default: sample_size is decided internally
            coreset_by: function, label, or list of labels, optional
                split each dataset by key
        """
        pass

    def partial_build_from_tensor(
            self,
            dataset: Any,
            *,
            sample_size: int = None,
            coreset_by: Union[Callable, str, list] = None
    ):
        """
        Applying new samples to a coreset tree based on the torch.Tensor

        Parameters:
            dataset: torch.Tensor
            sample_size: nodes are created based on input chunks
                default: sample_size is decided internally
            coreset_by: function, label, or list of labels, optional
                split each dataset by key
        """
        pass

    def save(
            self,
            dir_path: Union[str, os.PathLike] = None,
            name: str = None,
            save_buffer: bool = True,
            override: bool = False
    ) -> pathlib.Path:
        """
        Save service configuration and relevant data to a local directory.
        Use this method when the service needs to restored.

        Parameters:
            dir_path: string or PathLike, optional, default self.working_directory
                A local directory for saving service's files.

            name: string, optional, default service class name (lower case)
                Name of the sub-directory where the data will be stored.

            save_buffer: boolean, default True
                Save the service’s configuration and relevant data to a local directory.

            override: bool, optional, default false
                False: add a timestamp suffix so each save won’t override previous ones.
                True: existing sub-directory with that name is overridden.

        Returns:
            Save directory path.
        """
        pass

    @classmethod
    def load(
            cls,
            dir_path: Union[str, os.PathLike],
            name: str = None,
            *,
            data_manager: DataManagerT = None,
            load_buffer: bool = True,
            working_directory: Union[str, os.PathLike] = None
    ) -> 'CoresetTreeService':
        """
        Restore a service object from a local directory.

        Parameters:
            dir_path: str, path
                Local directory where service data is stored.

            name: string, optional, default service class name (lower case)
                The name prefix of the sub-directory to load.
                When more than one sub-directories having the same name prefix are found, the last one, ordered by name, is selected.
                For example when saving with override=False, the chosen sub-directory is the last saved.

            data_manager: DataManagerBase subclass, optional
                When specified, input data manger will be used instead of restoring it from the saved configuration.

            load_buffer: boole, optional, default True
                If set, load saved buffer from disk and add it to the tree.

            working_directory: str, path, optional, default use working_directory from saved configuration
                Local directory where intermediate data is stored.

        Returns:
            CoresetTreeService object
        """

        pass

    def get_coreset(self, level=0, as_orig=False, with_index=False) -> dict:
        """
        Get tree's coreset data either in a processed format or in the original format.
        Use the level parameter to control the level of the tree from which samples will be returned.

        Parameters:
            level: int, optional, default 0
                Defines the depth level of the tree from which the coreset is extracted.
                Level 0 returns the coreset from the head of the tree with up to coreset_size samples.
                Level 1 returns the coreset from the level below the head of the tree with up to 2* coreset_size samples, etc.

            as_orig: boolean, optional, default False
                Should the data be returned in it's original format or as a tuple of indices, X, and optionally y.
                True: data is returned as a pandas DataFrame.
                False: return a tuple of (indices, X, y) if target was used and (indices, X) when there is no target.

            with_index: boolean, optional, default False
                Relevant only when as_orig=True. Should the returned data include the index column.

        Returns:
            dict: data: numpy arrays tuple (indices,X, optional y) or a pandas DataFrame
                w: A numpy array of sample weights
                n_represents: number of instances represented by the coreset

        """
        pass

    def save_coreset(self, file_path, level=0, as_orig=False, with_index=False):
        """
        Get coreset from the tree and save to a file along with coreset weights.
        Use the level parameter to control the level of the tree from which samples will be returned.

        Parameters:
            file_path: string or PathLike
                Local file path to store the coreset.

            level: int, optional, default 0
                Defines the depth level of the tree from which the coreset is extracted.
                Level 0 returns the coreset from the head of the tree with up to coreset_size samples.
                Level 1 returns the coreset from the level below the head of the tree with up to 2* coreset_size samples, etc.


            as_orig: boolean, optional, default False
                True: save in the original format.
                False: save in a processed format (indices, X, y, weight).

            with_index: boolean, optional, default False
                Relevant only when as_orig=True. Save also index column.

        """
        pass

    def get_important_samples(
            self,
            size: int = None,
            class_size: Dict[Any, Union[int, str]] = None,
            ignore_indices: Iterable = None,
            select_from_indices: Iterable = None,
            select_from_function: Callable[[Iterable, Iterable, Iterable], Iterable[Any]] = None,
            ignore_seen_samples: bool = True,
            verbose: bool = False,

    ) -> Union[ValueError, dict]:
        """
        Returns indices of most important samples order by importance. Useful for identifying miss-labeled instances.
        At least one of size, class_size must be provided. Must be called after build.

        Parameters:
            size: int, optional
                Number of samples to return.
                When class_size is provided, remaining samples are taken from classes not appearing in class_size dictionary.

            class_size: dict {class: int or "all" or "any"}, optional.
                Controls the number of samples to choose for each class.
                int: return at most size
                "all": return all samples.
                "any": limits the returned samples to the specified classes.

            ignore_indices: array-like, optional.
                An array of indices to ignore when selecting important samples.

            select_from_indices: array-like, optional.
                 An array of indices to consider when selecting important samples.

            select_from_function: array-like, optional.
                 Pass a function in order to filter out unwanted samples. The function should accept 3 parameters
                 as input: indices, X, y and return a list(iterator) of  desired indices.

            ignore_seen_samples: bool, optional, default False.
                 Exclude already seen indices and set seen flag on any returned indices.

        Returns:
            Dict:
                indices: array-like[int].
                    important samples indices.
                X: array-like[int].
                    X array
                y: array-like[int].
                    y array
                importance: array-like[float].
                    The important value. High value is more important.

        Examples
        -------
        Input:
            size=100,
            class_size={"class A": 10, "class B": 50, "class C": "all"}
        Output:
            10 of "class A",
            50 of "class B",
            12 of "class C" (all),
            28 of "class D/E"
        """
        pass

    def set_seen_indication(self,
                            seen_flag: bool = True,
                            indices: Iterable = None,
                            ):
        """
        Set samples as 'seen' or 'unseen'. Not providing an indices list defaults to setting the flag on all
        samples.

        Parameters:
            seen_flag: bool
                Set 'seen' or 'unseen' flag
            indices: array like
                Set flag only on provided list of indices. Defaults to all indices.
        """
        pass

    @telemetry
    def remove_samples(
            self,
            indices: Iterable,
            force_resample_all: int = None,
            force_sensitivity_recalc: int = None,
            force_do_nothing: bool = False
    ):
        '''
        Remove samples from the coreset tree.

        Parameters:
            indices (Iterable): List of samples to be removed.
            force_resample_all (int): Force full resample for affected nodes, starting from level=force_resample_all, assuming:
                0 - root level;
                len(tree) = leaf level;
                None - no force resample at all;
                -1 - same as leaf level.
            force_sensitivity_recalc:
                Partial resampling for affected nodes, based on coreset quality,
                starting from  level=force_sensitivity_recalc, assuming:
                    0 - root level;
                    len(tree) - leaf level;
                    None - one level above leaf (same as force_sensitivity_recalc=len(tree)-1, default behaviour);
                    -1 - same as leaf level
            force_do_nothing: bool, default False
                If set True - do not make either full or partial resampling
        '''
        pass

    @telemetry
    def update_targets(
            self,
            indices: Iterable,
            y: Iterable,
            force_resample_all: int = None,
            force_sensitivity_recalc: int = None,
            force_do_nothing: bool = False
    ):
        """
        Update targets for samples on the coreset tree.

        Parameters:
            indices: Iterable
                List of samples to be removed.
            y:
                List of targets, should have the same length as indices
            force_resample_all:
                Force full resample for affected nodes, starting from level=force_resample_all, assuming:
                    0 - root level;
                    len(tree) = leaf level;
                    None - no force resample at all;
                    -1 - same as leaf level.
            force_sensitivity_recalc:
                Partial resampling for affected nodes, based on coreset quality,
                starting from  level=force_sensitivity_recalc, assuming:
                    0 - root level;
                    len(tree) - leaf level;
                    None - one level above leaf (same as force_sensitivity_recalc=len(tree)-1, default behaviour);
                    -1 - same as leaf level
            force_do_nothing: bool, default False
                If set True - do not make either full or partial resampling
        """
        pass

    @telemetry
    def update_features(
            self,
            indices: Iterable,
            X: Iterable,
            feature_names: Iterable[str] = None,
            force_resample_all: int = None,
            force_sensitivity_recalc: int = None,
            force_do_nothing: bool = False
    ):
        """
        Update features for samples on the coreset tree.

        Parameters:
            indices: List of samples to be removed.
            X: List of features, should have the same length as indices
            feature_names:
                If the quantity of features in X is not equal to the quantity of features in the original coreset,
                this param should contain list of names of passed features.
            force_resample_all:
                Force full resample for affected nodes, starting from level=force_resample_all, assuming:
                    0 - root level;
                    len(tree) = leaf level;
                    None - no force resample at all;
                    -1 - same as leaf level.
            force_sensitivity_recalc:
                Partial resampling for affected nodes, based on coreset quality,
                starting from  level=force_sensitivity_recalc, assuming:
                    0 - root level;
                    len(tree) - leaf level;
                    None - one level above leaf (same as force_sensitivity_recalc=len(tree)-1, default behaviour);
                    -1 - same as leaf level
            force_do_nothing: bool, default False
                If set True - do not make either full or partial resampling
        """
        pass

    @telemetry
    def apply_data_filter(
            self,
            filter_function: Callable[[Iterable, Iterable, Iterable], Iterable[bool]],
            force_resample_all: int = None,
            force_sensitivity_recalc: int = None,
            force_do_nothing: bool = False
    ):
        """
        Remove samples from the coreset tree, on the base of filter function,
        that returns list of samples to be *left* in tree. The signature is *filter_function(indexes, X, y)*.
        For example, for removing all instances for which target equal 6,
        the following function could be defined *filter_function = lambda indexes, X, y : indexes[y != 6]*.

        Parameters:
            filter_function:
                Function that returns list of samples to be left in tree. Its logic described above.
            force_resample_all:
                Force full resample for affected nodes, starting from level=force_resample_all, assuming:
                    0 - root level;
                    len(tree) = leaf level;
                    None - no force resample at all;
                    -1 - same as leaf level.
            force_sensitivity_recalc:
                Partial resampling for affected nodes, based on coreset quality,
                starting from  level=force_sensitivity_recalc, assuming:
                    0 - root level;
                    len(tree) - leaf level;
                    None - one level above leaf (same as force_sensitivity_recalc=len(tree)-1, default behaviour);
                    -1 - same as leaf level
            force_do_nothing: bool, default False
                If set True - do not make either full or partial resampling
        """
        pass

    def update_dirty(
            self,
            force_resample_all: int = None,
            force_sensitivity_recalc: int = None
    ):
        """
        Execute resampling on *dirty* nodes of the coreset tree, that is all nodes, that were affected with
        any of methods: *remove_samples, update_targets, update_features, apply_data_filter*, and were
        not partially or fully resampled.

        Parameters:
            force_resample_all:
                Force full resample for affected nodes, starting from level=force_resample_all, assuming:
                    0 - root level;
                    len(tree) = leaf level;
                    None - no force resample at all;
                    -1 - same as leaf level.
            force_sensitivity_recalc:
                Partial resampling for affected nodes, based on coreset quality,
                starting from  level=force_sensitivity_recalc, assuming:
                    0 - root level;
                    len(tree) - leaf level;
                    None - one level above leaf (same as force_sensitivity_recalc=len(tree)-1, default behaviour);
                    -1 - same as leaf level
        """
        pass

    def is_dirty(self) -> bool:
        """
        Returns a flag whether the coreset tree has "dirty" nodes, that is all nodes, that were affected with
        any of methods: remove_samples, update_targets, update_features, apply_data_filter, and were
        not resampled.
        """
        if not self.tree:
            return False
        else:
            return self.tree.is_dirty()

    @telemetry
    def fit(self, level=0, model=None, **model_params):
        """
        Fit a model on the coreset tree.
        Parameters:
            level: Defines the depth level of the tree from which the coreset is extracted.
                Level 0 returns the coreset from the head of the tree with up to coreset_size samples.
                Level 1 returns the coreset from the level below the head of the tree with up to 2* coreset_size samples, etc.

            model: an ml model instance, optional
                When provided, model_params are not relevant.
                Default: instantiate the service model class using input model_params.

            model_params: model hyper parameters kwargs
                Input when instantiating default model class.
        """
        pass

    @telemetry
    def predict(self, X):
        """
        Run prediction on the trained model.
        Parameters:
            X: an array of features
        Returns:
            Model prediction results
        """
        pass

    def predict_proba(self, X):
        pass

    def print(self):
        """
        Print the tree's string representation.
        """
        pass

    def plot(self, dir_path: Union[str, os.PathLike] = None, name: str = None) -> pathlib.Path:
        """
        Produce a tree graph plot and save figure as a local png file.
        Parameters:
            dir_path: string or PathLike
                Path to save the plot figure in; if not provided, or if isn't valid/doesn't exist, the figure will be saved in the current directory (from which this method is called).
            name: string, optional
                Name of the image file
        Returns:
            Image file path
        """
        pass

    def explain(self, X, model_scoring_function: Callable[[np.ndarray, Any], float]) -> Iterator[Tuple[Union[list, dict], str, str]]:
        """
        return leaf metadata and explainability path,
        using provided unlabeled examples and model scoring function.
        Parameters:
            X: array like
                unclassified samples

            model_scoring_function: callable[[array like, any], float]
                model scoring function which gets the X and the node's train model as params and returns a score in
                the range of [0,1]; this function drives the building of the explainability path.
        Returns:
            An iterator of (metadata,explanation) tuples:
                metadata:
                    selected leaf's metadata
                explanation:
                    free text explaining the built explainability path
        """
        pass




class CoresetTreeServiceSVD(CoresetTreeService):
    """Subclass of CoresetTreeService for SVD"""

    coreset_cls = CoresetSVD
    model_cls = WSVD
    coreset_params_cls = CoresetParamsSVD


class CoresetTreeServiceLG(CoresetTreeService):
    """Subclass of CoresetTreeService for Logistic Regression"""

    coreset_cls = CoresetLG
    model_cls = LogisticRegression
    coreset_params_cls = CoresetParamsLG


class CoresetTreeServicePCA(CoresetTreeService):
    """Subclass of CoresetTreeService for PCA"""

    coreset_cls = CoresetPCA
    model_cls = WPCA
    coreset_params_cls = CoresetParamsPCA


class CoresetTreeServiceKMeans(CoresetTreeService):
    """Subclass of CoresetTreeService for KMeans"""

    coreset_cls = CoresetKMeans
    model_cls = KMeans
    coreset_params_cls = CoresetParamsKMeans

    def _fit_internal(self, X, y, weights, model=None, n_clusters=None, **model_params):
        initial_centers, _ = kmeans_plusplus_w(X=X, n_clusters=n_clusters, w=weights, random_state=model_params.get('random_state'))
        model = self.model_cls(n_clusters=n_clusters, n_init=1, **model_params) if model is None else model
        model.set_params(init=initial_centers)
        model.fit(X, y, weights)
        return model


class CoresetTreeServiceLR(CoresetTreeService):
    """Subclass of CoresetTreeService for linear regression"""

    coreset_cls = CoresetReg
    model_cls = LinearRegression
    coreset_params_cls = CoresetParamsLR
